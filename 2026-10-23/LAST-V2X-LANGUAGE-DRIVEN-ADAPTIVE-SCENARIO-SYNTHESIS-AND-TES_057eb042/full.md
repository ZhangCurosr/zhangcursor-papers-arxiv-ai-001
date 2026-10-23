# LAST-V2X: LANGUAGE-DRIVEN ADAPTIVE SCENARIO SYNTHESIS AND TESTING FOR COMMUNICATION-INDUCED FAILURES IN V2X SAFETY APPLICATIONS

Keshu Wu<sup>1,2</sup>, Hao Zhang<sup>1</sup>, Rui Gan<sup>2</sup>, Pei Li<sup>3</sup>, Jiwan Jiang<sup>1</sup>, Yang Zhou<sup>1,∗</sup>

<sup>1</sup>Texas A&M University, <sup>2</sup>University of Wisconsin-Madison, <sup>3</sup>University of Wyoming

## ABSTRACT

Vehicle-to-everything (V2X) communication enables connected vehicles to make safety-critical decisions, but delayed, missing, or incorrect information can compromise driving safety. Systematically testing such failures is challenging because traffic interactions, communication conditions, and application responses must be jointly configured to produce meaningful scenarios. This paper presents LAST-V2X, a language-driven, adaptive framework for automated scenario synthesis and safety testing of V2X applications. LAST-V2X leverages large language models (LLMs) to translate natural-language test descriptions into executable scenarios grounded in roadway geometry, V2X communication semantics, and vehicle behavior. The framework integrates physical traffic simulation with spatially aware communication modeling and message-level fault injection to evaluate how information degradation affects vehicle decisions. To establish whether an observed safety failure is caused by communication rather than the underlying traffic conditions, LAST-V2X employs matched executions under degraded and ground-truth information. Furthermore, an execution-guided refinement mechanism combines parameter search and LLM self-reflection to identify ineffective test configurations, improve scenario effectiveness, and produce minimal reproducible failure cases. Experiments across diverse CARLA environments and V2X safety applications demonstrate the framework’s ability to generate reliable scenarios, distinguish communication-induced failures from unrelated safety violations, and substantially improve failure discovery through adaptive refinement. These results establish LAST-V2X as a systematic approach to language-driven, causally verified, and adaptive safety testing for connected vehicle systems.

Keywords V2X · connected vehicles · scenario generation · large language models · fault injection · simulation-based testing

3 Automated V2X Scenario Synthesis  
![](images/968436d04ee70303bacb83be83116f5f53afb1defe4d7d83189755fda39b7bfe.jpg)  
4 Scenario Refinement  
Overview of LAST-V2X<sup>1</sup>. LLM-driven synthesis of V2X test scenarios with causal checks and self-reflective refinement.

## 1 Introduction

Vehicle-to-everything (V2X) communication extends a vehicle’s awareness beyond onboard sensors [1–5]. Roadside units (RSUs) broadcast Signal Phase and Timing (SPaT) and intersection geometry (MAP) messages, while connected vehicles exchange Basic Safety Messages (BSMs). Applications such as Red-Light Violation Warning (RLVW), Intersection Movement Assist (IMA), and Left Turn Assist (LTA) use this information to support safety-critical decisions [6–8]. This communication–driving coupling also creates a cyber–physical safety pathway: delayed, missing, or incorrect V2X information can alter vehicle decisions and ultimately compromise safety. Systematically testing such communication-induced failures is therefore increasingly important. Constructing these tests manually is difficult because a meaningful V2X scenario must coordinate roadway, traffic, vehicle behavior, message type, communication condition, and fault timing. These elements are strongly coupled: a delayed SPaT message, for example, matters only when it occurs at the right movement, phase transition, and decision time. The resulting scenario space is large and highly structured, and many seemingly reasonable combinations are infeasible or ineffective.

Beyond the difficulty of specifying coupled V2X scenarios, constructing them in high-fidelity simulators can itself require substantial effort to understand simulator APIs, scenario logic, traffic control, and communication interfaces. Large language models (LLMs) [9, 10] offer a natural bridge between engineers’ high-level safety intents and the complex simulation logic required to instantiate executable V2X test scenarios. Transportation engineers can naturally express safety concerns in language. An LLM can translate such intent into structured requirements, identify the relevant safety application, connect semantic traffic roles with domain concepts, and retrieve the roadway, communication, and behavioral knowledge needed to instantiate the test [11–15]. Recent language-driven methods have demonstrated the potential of compiling natural-language descriptions into executable driving scenes [16–26], a direction recently surveyed in [27]. This capability is particularly valuable for V2X, where a scenario depends not only on how vehicles move, but also on what information they exchange and what each vehicle believes about the traffic state. LLMs can therefore reduce both scenario-engineering effort and simulator-programming burden by orchestrating heterogeneous transportation, communication, and simulation knowledge.

Despite this promise, generating a plausible V2X scenario does not guarantee a meaningful safety test. A communication fault matters only if it changes the information used by the V2X application and, in turn, alters the vehicle’s physical behavior or safety outcome. This cyber–physical effect can be difficult to establish because communication failures take different forms, including packet loss, latency, timestamp errors, decoding delays, frozen controller messages, position errors, and incorrect lane-to-signal-group associations [28–32]. Some missing or delayed messages become stale and can be detected by the application, potentially triggering a safe fallback. In contrast, incorrect information may still appear fresh and therefore be trusted, leading the vehicle to act on a wrong traffic state. Safety-critical information may also be absent entirely, for example when a conflicting vehicle does not transmit V2X messages. Therefore, language-driven generation alone is insufficient: the generated scenario must explicitly model how communication faults affect application decisions and verify whether those effects actually propagate into physical safety consequences.

Existing safety-critical scenario-generation methods primarily create risk by modifying the physical world [33–38]: vehicle trajectories, behaviors, road conditions, or initial states are perturbed until an unsafe interaction emerges [39–42]. Recent language-driven approaches further allow such physical scenarios to be specified and generated from natural-language descriptions [17, 20, 43–52]. V2X safety testing, however, poses a fundamentally different problem. The physical traffic scene may remain unchanged while communication causes the ego vehicle to perceive a different state of that scene. A signal may physically turn red while the received SPaT still indicates green; a conflicting vehicle may physically exist while its BSM is missing; or the roadway geometry may be correct while the received MAP associates the ego with the wrong movement. The safety-critical condition therefore arises not simply from what happens in the physical world, but from a mismatch between the physical world and the information available to the vehicle.

V2X co-simulation and digital-twin platforms provide mechanisms for modeling message exchange, latency, packet loss, and communication-dependent vehicle control [29, 53–58]. Automated testing of V2X systems has so far focused mainly on cooperative perception, for example by transforming shared sensor data under varied communication and weather conditions [59], rather than on message-level faults in closed-loop safety applications. However, communication simulation alone does not solve the testing problem. The physical interaction, message semantics, fault type, fault timing, and application response must be jointly constructed so that the information mismatch occurs precisely when it can affect a safety-critical decision. A generated test may also be executable yet ineffective because the selected fault is mistimed, too weak, or semantically inconsistent with the intended failure mechanism. Existing language-driven methods largely stop after scenario generation or, when they close the loop, refine only the physical adversarial behavior [46], while communication simulators typically assume that the scenario and fault configuration are already specified; neither provides a mechanism to use execution evidence to diagnose why a test failed and revise the underlying test intent. Moreover, observing a collision after injecting a communication fault is insufficient to establish a V2X-induced failure, because the same collision may occur even under healthy communication. Existing language-driven scenario generation, communication simulation, and safety falsification therefore address different parts of the problem, but do not provide an integrated, self-refining framework for constructing, diagnosing, refining, and causally validating failures that originate in the information layer and propagate into the physical world. This leads to the central question of this work: can natural-language V2X test intents be transformed into executable scenarios that systematically expose and causally verify communication-induced safety failures?

![](images/da3a91623db260ce1add25fbaf2ec8718d2a9d4ed4202fe2df4281804d3ecc42.jpg)  
Figure 1: Overview of LAST-V2X. LLM Agent Analysis verifies the intent; Retrieval-Augmented Scenario Grounding places it on a feasible junction; Automated V2X Scenario Synthesis runs it in CARLA with faulted messages beside a ground-truth ego; Scenario Refinement admits a failure only if the message-driven ego alone fails, else refines it.

To address this question, we develop LAST-V2X, an LLM-orchestrated framework for communication-aware V2X safety testing. LAST-V2X first converts natural-language intents into grounded scenario specifications, then executes coupled traffic–communication scenarios in CARLA under matched degraded and non-degraded communication conditions. Execution evidence is further used to refine ineffective scenarios through parameter search and LLM self-reflection, while successful failures are reduced to minimal reproducible cases.

## The main contributions are:

1. Coupled physical–information scenario generation. LAST-V2X jointly constructs the physical traffic scene and the V2X information perceived by the vehicle, with feasibility checks to reject ineffective tests before simulation.

2. Causal verification of communication-induced safety failures. Matched degraded and non-degraded executions determine whether V2X faults actually cause changes in physical safety outcomes.

3. Execution-grounded adaptive refinement. Parameter search corrects quantitative deficiencies, while LLM self-reflection revises semantically ineffective test intents using simulation evidence. Successful cases are further reduced to minimal reproducible failures.

The remainder of this paper is organized as follows. Section 2 formulates the testing problem and introduces the four modules of LAST-V2X. Section 3 presents the experimental evaluation, and Section 4 concludes the paper.

## 2 Methodology

As illustrated in Fig. 1, LAST-V2X contains four interdependent modules. LLM Agent Analysis converts the user intent into a typed and feasibility-verified test specification. Retrieval-Augmented Scenario Grounding binds the symbolic specification to roadway geometry, V2X domain knowledge, and executable agent behaviors. Automated V2X Scenario Synthesis realizes the resulting program as a coupled physical–information simulation with spatial communication, semantic message faults, and paired ego rollouts. Finally, Scenario Refinement uses execution evidence to search for, diagnose, and minimize communication-induced failures. We first formalize the problem solved by these four modules.

## 2.1 Problem Description

Scenario Specification The input to LAST-V2X is a short free-form natural-language prompt p describing a traffic situation to be tested. The prompt may specify the environment, ego maneuver, surrounding vehicles, safety-critical interaction, and optionally a communication condition or fault. A compiler C transforms the prompt into a structured scenario specification

$$
S = { \mathcal { C } } ( p ) = { \big ( } a , E , Q , \mathcal { V } , u _ { \mathrm { e g o } } , C , F { \big ) } ,\tag{1}
$$

where a denotes the V2X safety application under test; $E$ describes environmental conditions; Q specifies geometric and traffic-control requirements; V is the roster of surrounding agents; $u _ { \mathrm { e g o } }$ defines the ego maneuver and initial conditions; $C$ describes the nominal communication configuration; and $F = \overbrace { \{ F _ { 1 } , \ldots , F _ { m } \} } ^ { - \infty }$ denotes the set of communication faults.

Each non-ego vehicle $i \in \nu$ is characterized by its vehicle class, semantic role in the interaction, initial distance and speed, V2X equipage status, and behavioral character. The communication configuration includes message broadcast rates, communication parameters, V2X penetration, and the signal plan Ψ. Each fault specifies a message-level operator, its target information item, an event-anchored activation window, and its associated magnitude or parameters.

Physical and Perceived States Each scenario exercises one V2X application a, represented as a decision function

$$
\pi _ { a } ( \beta , x ) \to u ,\tag{2}
$$

where x is the ego physical state, $\beta$ is the information state reconstructed from received V2X messages, and u is the resulting control action. Importantly, the application does not act directly on the simulator’s ground-truth world state. Its behavior depends on what information has been sampled, communicated, possibly degraded, and reconstructed at the receiver.

We represent this transformation through the information chain

$$
W \ { \overset { \Sigma } { \longrightarrow } } \ { \mathcal { M } } \ { \overset { \Gamma } { \longrightarrow } } \ { \mathcal { M } } ^ { \prime } \ { \overset { \Phi _ { F } } { \longrightarrow } } \ { \mathcal { M } } ^ { \prime \prime } \ { \overset { \ B } { \longrightarrow } } \ \beta ,\tag{3}
$$

where $W ( t )$ is the ground-truth world state, Σ is message sampling, Γ is communication-channel delivery, $\Phi _ { F }$ is fault injection, and $\boldsymbol { B }$ is receiver-side state reconstruction. Thus, $\beta ( t )$ is generally only an imperfect representation of $W ( t )$

A generated message is represented by

$$
m _ { k } = ( t _ { k } , \hat { t } _ { k } , y _ { k } ) ,\tag{4}
$$

where $t _ { k }$ is its true generation time, $\hat { t } _ { k }$ is the timestamp reported by the message, and $y _ { k }$ is its semantic payload. For information item $i ,$ the receiver retains the most recent accepted payload,

$$
\beta _ { i } ( t ) = y _ { k ^ { * } , i } ,\tag{5}
$$

whose Age of Information is

$$
\Delta _ { i } ( t ) = t - \hat { t } _ { k ^ { * } } .\tag{6}
$$

This representation distinguishes three information-failure mechanisms used throughout the paper. In a stale-butdetectable condition, loss or delay increases $\Delta _ { i } ( t )$ and can trigger a freshness-based fallback. In a fresh-but-wrong condition, messages remain apparently current while their semantic content differs from the physical world. A silentby-absence condition occurs when safety-relevant information is never generated, for example because a conflicting vehicle is not V2X-equipped.

Causal Failure Objective A difference between $\beta ( t )$ and $W ( t )$ is not itself a safety failure, and an observed collision is not necessarily caused by communication. The objective of LAST-V2X is therefore to discover scenarios in which degraded V2X information changes the safety outcome.

For an executed scenario S, let $V _ { \mathrm { m s g } } ( S ) \in \{ 0 , 1 \}$ indicate whether the message-driven ego violates the applicationspecific safety criterion, and let $V _ { \mathrm { g t } } ( \breve { S } ) \in \{ 0 , \bar { 1 } \}$ denote the corresponding outcome when the same application acts on ground-truth information. We define the causal-admission indicator

$$
A ( S ) = \mathbb { I } \left[ V _ { \mathrm { m s g } } ( S ) = 1 \ \wedge \ V _ { \mathrm { g t } } ( S ) = 0 \right] .\tag{7}
$$

A scenario is therefore admitted as a communication-induced failure only when the message-driven ego fails while its matched ground-truth control remains safe.

The primary falsification objective is

$$
\operatorname* { m a x } _ { S \in { \mathcal { F } } } A ( S ) ,\tag{8}
$$

where $\mathcal { F }$ denotes the set of executable specifications satisfying the schema, analytic feasibility conditions, roadway constraints, and applicable intent-preservation requirements.

Since $A ( S )$ is binary and failures can be sparse, the search is guided before admission by a continuous failure margin $m ( S )$

$$
\operatorname* { m i n } _ { S ^ { \prime } \in \mathcal { N } ( S ) \cap \mathcal { F } } m ( S ^ { \prime } ) ,\tag{9}
$$

where $\mathcal { N } ( S )$ denotes bounded variations of the current scenario. Smaller margins indicate scenarios closer to violating the target safety criterion and are therefore prioritized during refinement. Appendix E defines $m ( S )$ and the search settings.

Once an admitted failure has been identified, the objective changes from discovery to explanation. Let $F ( S )$ denote its injected fault set. The minimal-witness problem is

$$
F ^ { * } = \operatorname * { a r g m i n } _ { F ^ { \prime } \subseteq F ( S ) } | F ^ { \prime } | \qquad \mathrm { s . t . } \qquad A ( S [ F ^ { \prime } ] ) = 1 ,\tag{10}
$$

so the final artifact retains only the smallest fault configuration required to reproduce the communication-caused failure; for a silent-by-absence failure, $F ( S ) = \emptyset$ and the witness is the unfaulted scenario itself. At the benchmark level, multiple such witnesses are sought across different applications, fault classes, penetration levels, and conflict geometries rather than repeatedly maximizing the severity of a single mechanism.

The four modules below implement this process from language interpretation to causally verified failure discovery.

## 2.2 Module I: LLM Agent Analysis

The first module converts an open-ended natural-language request into a structured V2X test that is both syntactically valid and analytically capable of exercising the intended communication mechanism. It combines a schema-constrained LLM semantic parser with a deterministic feasibility verifier. The LLM determines what the prompt means; explicit rules and arithmetic determine whether that interpretation can produce a meaningful test.

Schema-Constrained Intent Parsing Let Λ denote the language interpretation operator. Given prompt $p ,$ it produces an initial candidate specification

$$
S ^ { ( 0 ) } = \Lambda ( p ) .\tag{11}
$$

The parser uses structured JSON output constrained by the schema of Eq. (1). It identifies the application under test, environment and map requirements, ego and surrounding-agent attributes, communication configuration, and requested faults. When the prompt does not explicitly name the safety application, Λ infers it from the traffic interaction. Explicitly specified communication conditions are preserved, whereas unspecified fields are filled from predefined nominal defaults rather than through unconstrained generation.

Parsing is implemented as a bounded repair loop. If a candidate specification violates the schema or fails the analytic verifier, its diagnostic feedback is returned to the same interpretation operator:

$$
S ^ { ( k + 1 ) } = \Lambda \left( p , S ^ { ( k ) } , { \mathcal { E } } ( S ^ { ( k ) } ) \right) ,\tag{12}
$$

where $\mathcal { E } ( S ^ { ( k ) } )$ contains machine-readable violations together with actionable repair guidance. The LLM therefore performs semantic interpretation and repair, while the admissible specification space remains controlled by explicit constraints. Appendix F gives the parser, planner, and reflection prompts in condensed form.

Analytic Feasibility Verification The verifier determines whether a candidate can realize the communication condition it claims to test before simulator resources are committed. For a temporal fault $F _ { j }$ , realizability is evaluated against the freshness budget $B _ { a }$ of the information stream consumed by application a. A fault is potentially consequential when

$$
\overline { { { \Delta } } } ( F _ { j } ) > B _ { a } ,\tag{13}
$$

where $\overline { { \Delta } } ( F _ { j } )$ is a lower bound on the peak Age of Information (AoI) [60] induced by the fault. For a stream with broadcast period $T _ { s }$ and nominal latency L, dropping k consecutive messages gives

$$
\overline { { \Delta } } = k T _ { s } + L ,\tag{14}
$$

whereas an additional communication delay d gives

$$
{ \overline { { \Delta } } } = L + d .\tag{15}
$$

The true peak can exceed these bounds by up to one broadcast period, since that much time may already have passed since the last delivery; a fault that satisfies $\overline { { \Delta } } ( F _ { j } ) > B _ { a }$ is therefore guaranteed to breach the budget. If a requested drop is too weak, for example, the verifier can return the smallest count whose bound reaches the budget

$$
k ^ { * } = \left\lceil \frac { B _ { a } - L } { T _ { s } } \right\rceil .\tag{16}
$$

Semantic-content faults require a different test because they can remain apparently fresh. For a frozen SPaT value, incorrect MAP binding, or perturbed BSM position, the verifier checks whether the targeted information item is actually consumed by application a, whether the activation window overlaps a relevant decision period, and whether the requested corruption can produce $\dot { \beta } ( t ) \neq W ( t )$

Additional checks cover fault-window validity, conflict timing, V2X equipage consistency, nominal communication settings, and roadway satisfiability. Blocking violations are returned through E to the parser. The same verification gate is later reused to eliminate infeasible mutants during Scenario Refinement. Appendix A lists all check families.

## 2.3 Module II: Retrieval-Augmented Scenario Grounding

A verified specification describes what interaction should occur, but it is still symbolic. It does not yet identify where that interaction can physically occur, which concrete lanes correspond to semantic roles such as “crossing from the left,” or how the specified agents should behave. The second module therefore binds the specification to retrieved roadway, V2X, and behavioral knowledge [61] and produces a replayable scenario program.

Signal-Semantic Road Retrieval A signal-semantic road graph G is constructed offline from the OpenDRIVE map of each simulator town. Unlike a conventional road graph that records only coarse topology or whether an intersection is signalized, G preserves controller identity, traffic-signal identity, junction connections, controlled movements, and signal groups. These quantities are required not only for geometric grounding but also because SPaT and MAP faults directly operate on them.

The basic retrieval unit is

$$
\ell = ( { \mathrm { j u n c t i o n } } , { \mathrm { m o v e m e n t } } , { \mathrm { s i g n a l g r o u p } } , \mathbf { f } _ { \ell } ) ,
$$

where $\mathbf { f } _ { \ell }$ records geometric and traffic-control attributes. Given application a and map query $Q .$ , the candidate set is

$$
{ \mathcal { L } } ( S ) = \left\{ \ell \in { \mathcal { G } } : \operatorname { p r e } ( a ) ( \mathbf { f } _ { \ell } ) \wedge q _ { Q } ( \ell ) \right\} ,\tag{17}
$$

where $\mathrm { p r e } ( a )$ encodes the geometric preconditions of the target application and $q _ { Q }$ applies the map requirements inferred from the prompt. A seeded deterministic rule selects $\ell ^ { * }$ from the feasible set. If no candidate exists, the failed constraint is returned to Module I for repair.

V2X and Driver-Behavior Retrieval A second knowledge base contains the V2X semantics required to convert the traffic scene into a safety test. It represents the three message streams implemented by LAST-V2X: SPaT messages describing signal phase and timing, MAP messages describing intersection geometry and lane-to-signal-group associations, and BSMs describing the states of equipped vehicles.

An application registry maps each application a to the information streams it consumes, its geometric preconditions, and its information-quality budgets. A fault-pattern library maps natural-language communication degradations to the protocol-level fault operators implemented by the runtime.

Driver behavior is instantiated through parameterized templates. For vehicle i,

$$
\theta _ { i } = w _ { E } \odot T _ { \mathrm { b e h } } ( c _ { i } , b _ { i } ) ,\tag{18}
$$

where $T _ { \mathrm { b e h } }$ maps vehicle class $c _ { i }$ and behavioral character $b _ { i }$ to parameters including desired speed, time headway, standstill clearance, reaction time, comfortable deceleration, and signal compliance. Environmental weighting $w _ { E }$ adjusts these parameters according to E.

Lane-Level Scene Planning The retrieved movement unit determines a feasible junction, but each semantic agent role must still be mapped onto an actual lane. The executor queries the selected junction in the live simulator and constructs an admissible approach set $\mathcal { I } ( \ell ^ { * } )$ containing lane identifiers, movement directions, and measured headings relative to the ego approach.

A constrained planning operator assigns agents to these retrieved approaches:

$$
\mathcal { Z } = \Lambda _ { \mathrm { p l a n } } \left( S , \mathcal { T } ( \ell ^ { * } ) \right) .\tag{19}
$$

The LLM may therefore resolve semantic relationships such as “oncoming” or “crossing from the $\mathrm { l e f t } , \ ' $ but it cannot invent lanes or roadway geometry.

Each assignment is then validated deterministically. Using the specified spawn distance and initial speed, the validator checks whether the relevant vehicles can reach the conflict region within the required encounter window. Invalid assignments are rejected and deterministic grounding heuristics are used as a fallback where necessary.

Scenario Program Assembly Finally, the verified specification, retrieved movement, lane assignments ${ \mathcal { Z } } ,$ and behavior parameters are combined through

$$
P = { \mathrm { b i n d } } \left( S , \ell ^ { * } , \mathcal { Z } , \{ \theta _ { i } \} \right) .\tag{20}
$$

The resulting program contains the agent routes, signal plan Ψ, V2X message streams, communication settings, event-anchored fault schedule, and monitoring configuration required for execution.

Importantly, P is typed scenario data consumed by a fixed runtime engine; it is not LLM-generated simulator source code. This makes each test reproducible, allows later refinement operators to manipulate the specification symbolically, and preserves traceability between the executable scenario and the original prompt or retrieved knowledge.

## 2.4 Module III: Automated V2X Scenario Synthesis

The third module executes the grounded scenario program as a coupled physical–information process. Unlike a conventional traffic simulation that models vehicle motion alone, $\mathrm { L A S T - V } 2 X$ synchronously advances the traffic state, signal state, V2X messages, receiver-side estimated state, application decisions, and paired ego outcomes. Execution produces the synchronized trace

$$
{ \mathcal { T } } = \operatorname { E x e c } ( P ) .\tag{21}
$$

Synchronous Physical–Information Execution Simulation proceeds synchronously with fixed step $\Delta t = 5 0$ ms. At each tick, messages are sampled from the current world, propagated through the communication channel, degraded by any active faults, and incorporated into the estimated information state. The application then acts on that estimate:

$$
\beta _ { t + \Delta t } = B \left( \beta _ { t } , ( \Phi _ { F } \circ \Gamma \circ \Sigma ) ( W _ { t } ) \right) ,\tag{22}
$$

followed by

$$
W _ { t + \Delta t } = \Phi _ { \mathrm { s i m } } ( W _ { t } , \pi _ { a } ( \beta _ { t } , x _ { t } ^ { \mathrm { m s g } } ) , u _ { t } , \Psi _ { t } ) ,\tag{23}
$$

where $u _ { t }$ contains the controls of the non-ego agents.

Vehicle routes are instantiated on the live simulator map, initial conditions are stabilized before the critical encounter, conflict-relevant agents follow the scripted controllers parameterized by $\theta _ { i }$ , and background traffic is controlled through a seeded Traffic Manager. The signal plan is aligned with the intended encounter so that event-anchored communication faults occur relative to the corresponding traffic or signal event rather than at an arbitrary simulation time.

Spatial V2X Communication Channel The communication operator Γ models delivery as a function of the instantaneous physical relationship between transmitter and receiver. For each link, latency L and delivery probability $p _ { \mathrm { d e l } }$ are obtained from received signal strength:

$$
( L , p _ { \mathrm { d e l } } ) = g ( \mathrm { R S S I } ) , \qquad \mathrm { R S S I } = P _ { \mathrm { t x } } - 1 0 n \log _ { 1 0 } d - B _ { \mathrm { b l k } } ( x _ { \mathrm { t x } } , x _ { \mathrm { r x } } ) ,\tag{24}
$$

where $P _ { \mathrm { t x } }$ is transmit power, n is the path-loss exponent, d is transmitter–receiver distance, and $B _ { \mathrm { b l k } }$ is the heightaware building-blockage term. The mapping g increases latency and reduces delivery probability as signal strength deteriorates.

Roadside units are placed at physical signal-pole locations, so communication quality evolves with vehicle motion, urban geometry, and transmitter placement rather than being represented by one scenario-wide packet-loss rate. Appendix B gives the complete model and its default parameters.

Semantic V2X Fault Injection After nominal channel propagation, the fault engine $\Phi _ { F }$ applies protocol-level message faults. Seven operators are implemented across three failure families. Delivery faults include drop, burst\_drop, and delay; the timestamp fault skew changes the reported generation time; and semantic-content faults include freeze, rebind, and pos\_perturb. Appendix C defines each operator and its parameters.

These operators alter different parts of the information chain. Delivery and timing faults primarily change message availability or apparent age, whereas semantic faults may preserve normal delivery and fresh timestamps while changing the represented state. Fault windows are anchored to traffic or signal events and converted into deterministic simulation-time intervals, allowing identical tests to be reproduced across paired executions and subsequent replays.

Paired Counterfactual Execution A shadow ground-truth ego is propagated alongside the message-driven ego. Both use the same application, controller, route, initial condition, traffic scene, and simulation seed. Their only systematic difference is the information provided to the application: the message-driven ego acts on $\beta ( t )$ , whereas the ground-truth ego acts on the corresponding state derived directly from $W ( t )$

The shadow ego has no physical presence and therefore cannot alter surrounding traffic. This paired construction provides the counterfactual required by Eq. (7). Each simulation tick records physical state, signal state, message transmissions and receptions, reconstructed information, application decisions, and both ego outcomes. The resulting trace T therefore provides an auditable record of the complete chain from communication degradation to behaviora consequence.

## 2.5 Module IV: Scenario Refinement

A single execution may expose an admitted failure, but many initial tests are too weak, mistimed, or otherwise poorly configured to make the intended communication mechanism consequential. Scenario Refinement therefore closes the loop between simulation evidence and scenario generation.

Two complementary refinement processes are used. An inner specification-space search varies bounded properties of the current grounded scenario while preserving its semantic intent. An outer prompt-space reflection revises the natural-language test description when the execution evidence indicates that the original intent itself prevents the desired mechanism from becoming consequential. Al gorithm 1 summarizes both loops.

Execution Diagnosis and Causal Admission The refinement monitor first identifies intervals in which a safety-relevant estimated information item differs from its corresponding ground-truth state,

$$
\beta _ { i } ( t ) \neq W _ { i } ( t ) .
$$

Algorithm 1 Scenario refinement by search and reflection.   
Input: prompt $p ^ { ( 0 ) }$ , goal g, budgets $N _ { r } , N _ { s } ,$ elite size K   
Output: minimal witnesses, or diagnoses $\{ \boldsymbol { { \mathcal { D } } ^ { ( r ) } } \}$   
1: for $r = 0 , \ldots , N _ { r } - 1$ do ▷ reflection   
2: $S  { \mathcal { C } } ( p ^ { ( r ) } ) ; { \mathcal { T } }  \operatorname { E x e c } ( P )$ ▷ Mod. I–III   
3: if $A ( S ) { } ^ { * } = 1$ then return DELTADEBUG(S)   
4: ${ \mathcal { S } } \gets \{ { \mathcal { S } } \} ; { \mathcal { H } } \gets { \emptyset }$ ▷ elite, admitted   
5: for $b \doteq \dot { 1 } , \ldots , N _ { s }$ do ▷ search   
6: $S ^ { \prime }  \mathbf { M }$ UTATE $\left( \operatorname { S A M P L E } _ { w } ( S ) \right)$   
7: $\mathbf { i f } \neg \mathbf { V } \scriptscriptstyle \mathrm { E R I F Y } ( S ^ { \prime } )$ then continue ▷ gated   
8: execute $S ^ { \prime } { \ ; }$ compute $m ( S ^ { \prime } )$   
9: if $A ( S ^ { \prime } ) = 1$ then ${ \mathcal { H } }  { \dot { \mathcal { H } } } \cup \{ S ^ { \prime } \}$ ▷ catalog   
10: ${ \mathcal { S } }  \operatorname { T o p } _ { K } ( { \mathcal { S } } \cup \{ S ^ { \prime } \} )$ by w   
11: end for   
12: if $\cdot \varkappa \neq$ ∅ then return DELTADEBUG(H)   
13: $\mathcal { D } ^ { ( r ) } \gets \mathrm { D I G E S T } ( \mathcal { T } , \dots )$   
14: $p ^ { ( r + 1 ) }  \Lambda _ { \mathrm { r e f } } ( p ^ { ( r ) } , { \mathcal { D } } ^ { ( \dot { r } ) } , g )$ ▷ Eq. (27)   
15: end for   
16: return $\{ \mathcal { D } ^ { ( r ) } \} _ { r = 0 } ^ { N _ { r } - 1 }$ ▷ no admission

(25)

Each divergence episode is related to the next application decision through a hazard slack δ. Negative slack indicates that a decision was made while the estimated state was incorrect, whereas a small positive value denotes a near miss. Expected absences, such as missing BSMs from unequipped vehicles, are labeled separately as structural silent-by-absence conditions.

The paired rollout then evaluates the admission criterion of $\mathrm { E q . } ( 7 )$ . If both egos violate the safety criterion, the scenario is treated as non-attributable because the underlying traffic interaction itself is unsafe. If neither fails, the trace remains useful for determining how close the communication-induced divergence came to changing the outcome. Only

$$
V _ { \mathrm { m s g } } = 1 , \qquad V _ { \mathrm { g t } } = 0
$$

constitutes a communication-caused failure.

Specification-Space Search When the current specification is valid but does not produce an admitted failure, the inner loop searches a bounded neighborhood of that scenario. Candidate mutations modify parameters including fault type, magnitude and activation window, vehicle distance and speed, and V2X penetration. These changes preserve the high-level test intent while altering how strongly the information degradation overlaps the safety-critical interaction.

Every proposed mutant is first passed through the analytic verifier. Feasible mutants are executed and ranked using failure margin m(S). Candidates closer to failure receive greater probability of producing descendants according to

$$
w ( S ) = \frac { 1 } { 1 / 2 + m ( S ) } .\tag{26}
$$

Thus, execution-derived near misses focus subsequent search without replacing the causal admission test itself.

Search is also coverage-oriented. Admitted scenarios are cataloged by mechanism dimensions including application, fault family, V2X penetration, and conflict geometry, discouraging repeated discovery of equivalent failures.

Each admitted scenario is then minimized by delta debugging, a greedy descent: at every step it tries each single weakening of the fault vector (removing a fault, halving its magnitude, or shortening its window) and keeps the first that preserves admission under the same paired causal test, stopping when none does. The result approximates the minimal-witness objective of Eq. (10): it is 1-minimal, in that no single fault can be removed or weakened without losing the failure, and it yields a replayable scenario containing only the communication degradation necessary for the failure.

Prompt-Space Reflection Bounded specification mutation cannot repair every unsuccessful test. In some cases, the natural-language intent itself makes the requested failure mechanism irrelevant. For example, a prompt may activate a SPaT freeze only after the ego has already completed the safety-critical decision. Increasing the magnitude of that fault cannot make it causal without changing its temporal relationship to the described event.

The outer loop therefore constructs an evidence digest $\mathcal { D } ( \mathcal { T } ^ { ( r ) } )$ containing the paired outcomes, divergence episodes, hazard slack, decision-epoch AoI, failure margins, and verifier diagnostics. Following verbal self-reflection and self-refinement for LLM agents [62, 63], a dedicated reflection operator then updates the prompt:

$$
p ^ { ( r + 1 ) } = \Lambda _ { \mathrm { r e f } } \left( p ^ { ( r ) } , { \cal D } ( { \cal T } ^ { ( r ) } ) , g \right) .\tag{27}
$$

The purpose of reflection is to diagnose why the intended mechanism failed to affect the application and revise the test description accordingly. It may, for example, move a fault window so that it overlaps the decision, extend a semantic corruption through a signal transition, or modify a described geometry that prevents communication exposure.

The reflected prompt receives no privileged execution path. It re-enters Module I and is parsed and analytically verified, is grounded again through Module II, and is executed through Module III under the same causal admission criterion. Reflection therefore cannot directly manipulate simulator state or force a violation; it can only propose a revised test whose validity and causal effect must again be established by the complete pipeline.

## 3 Experimental Results

The experiments evaluate the three main capabilities of LAST-V2X: language-driven V2X scenario generation, causal verification of communication-induced failures, and adaptive refinement of ineffective tests. A 100-scene benchmark first evaluates whether natural-language scenarios can be generated reliably without producing false failures. Communication-fault experiments then test whether the paired execution can distinguish communication-induced failures from unsafe traffic interactions. Finally, a 50-scene refinement study evaluates whether parameter search and LLM self-reflection can systematically turn ineffective tests into causally verified failures. Ablation studies isolate the contribution of the key components.

## 3.1 Experiment Setup

Implementation and Protocol All experiments use the complete LAST-V2X pipeline with synchronous CARLA simulation across eight towns, except the specification-space search runs of the refinement study, which use a lightweight stub backend. GPT-4o [64] is used for language parsing, planning, and reflection with temperature 0.2 and a fixed sampling seed. Scenario seeds control vehicle equipage, communication loss, RSU selection, vehicle blueprints, and retrieval tie-breaking, enabling deterministic replay. For each scene, LAST-V2X records the generated specification, grounding results, execution traces, safety outcomes, and communication states. The evaluation includes 112 live CARLA scenes; 110 prompts compile on the first attempt and two require verifier-guided correction. Unless otherwise stated, results are reported from deterministic single-seed runs. Appendix G summarizes the configuration of each experiment.

b Driving Quality  
![](images/54e3a34b95ef9caf2b84bcbd47df3b70df211fabc8612f32d53ec012c8d79b8c.jpg)

![](images/50834d33da466e1289451af7b4e3564d8e7ac6fe2dc9af01c9593889fc60f1dd.jpg)

![](images/eb750328e86cb795db1340467cba532a9817649f9feb94c20432c86784cb927f.jpg)

![](images/3ef0ed8b6a490f9ce33077d3a950dec0631626af7d3f612a5fe9ead5c18d087b.jpg)

Table 1: V2X safety applications and their message requirements (SAE J2945/1 [65]). Age budgets bound the staleness of the consumed SPaT or BSM stream; “–” marks a quantity the application does not use.
<table><tr><td rowspan="2">Application Full name</td><td rowspan="2"></td><td rowspan="2">Function</td><td colspan="2">Age budget (s)</td><td rowspan="2">Position error (m)</td></tr><tr><td>SPaT</td><td>BSM</td></tr><tr><td colspan="6">Signal applications infrastructure-to-vehicle (SPaT/MAP)</td></tr><tr><td>RLVW</td><td>Red-Light Violation Warning</td><td>Warns against entering on red</td><td>1.0</td><td></td><td></td></tr><tr><td>GLOSA</td><td>Green Light Optimal Speed Advisory</td><td>Recommends speed for a green arrival</td><td>1.0</td><td></td><td></td></tr><tr><td colspan="6">Conflict applications vehicle-to-vehicle (BSM)</td></tr><tr><td>IMA</td><td>Intersection Movement Assist</td><td>Warns of crossing-vehicle conflicts</td><td></td><td>0.5</td><td>1.5</td></tr><tr><td>LTA</td><td>Left Turn Assist</td><td>Warns of conflicting opposing traffic</td><td>1.0</td><td>0.5</td><td>1.5</td></tr><tr><td>FCW</td><td>Forward Collision Warning</td><td>Warns of a slower or stopped lead vehicle</td><td></td><td>0.3</td><td>1.5</td></tr><tr><td>EEBL</td><td>Emergency Electronic Brake Light</td><td>Relays a hard-braking event ahead</td><td></td><td>0.3</td><td></td></tr></table>

Safety Applications We evaluate six V2X safety applications, summarized in Table 1. RLVW and GLOSA are signal applications that primarily use infrastructure-to-vehicle information, while IMA, LTA, FCW, and EEBL are conflict applications that primarily use vehicle-to-vehicle BSMs. This distinction captures two important V2X settings: infrastructure information can be shared by multiple vehicles, whereas a missing BSM can make a conflicting vehicle invisible to the receiving application. Appendix D gives each application’s decision rule and violation criterion.

e One Scene per Application  
![](images/34c2a328f5bf988d7154cff50bda820425e6e25a93705aeac6c76d3e9ad45eed.jpg)

![](images/b10e148518e8e621b33d6ecc21f3e13b33459ad671d48cc655787fb3d0770954.jpg)  
Figure 2: 100 generated scenes. a, Application and town. b, Driving quality (RF, route following; ACC, acceleration; YV, yaw rate; LI, lane invasion; CE, communication exposure). c, Scene conditions. d, Minimum gap. e, Examples.

Experiment Groups The evaluation contains three main experiment groups. First, a 100-scene normal-traffic benchmark tests whether natural-language descriptions can be converted into valid V2X scenarios without producing false communication-induced failures. The prompts describe only the traffic scene and environment; the exercised application, town, and nominal communication configuration are inferred or assigned by the pipeline. The benchmark spans all six applications and all eight CARLA towns, together with varied weather, times of day, and vehicle classes (Fig. 2a,c). Second, a communication-fault suite evaluates whether LAST-V2X can distinguish failures caused by degraded V2X information from those caused by the underlying traffic interaction. Third, the suite is extended to 50 scenarios to evaluate adaptive refinement through specification-space search and LLM self-reflection. Corridor, multi-CV, and channel-sensitivity experiments are treated as supplementary studies.

Evaluation Metrics Metrics are organized at four levels. The run level determines whether communication caused a safety failure. Each executed scenario is classified as ADMITTED, NON-ATTRIBUTABLE, or CLEAN. ADMITTED means the degraded-information rollout violates the safety criterion while the matched non-degraded rollout does not.

![](images/8bf49def528f9a4f90a3e0bbbff93b7e2b6c0f678998b1f4e23b248cea561737.jpg)

![](images/22207f4c34bdb04a6c8150873effd99aa3a8d0ac111581f262ba010dd7a95180.jpg)  
Figure 3: A nine-agent left turn in Town03 under a healthy channel. a, Prompt and compiled specification. b, Six instants from four viewpoints. c, Agent paths and V2I range. d, True and estimated signal phases. e, Message deliveries.

## a Scene Prompt

scene d13 · Town03

c Scene Layout

NON-ATTRIBUTABLE means both rollouts violate, indicating that the failure cannot be attributed to communication.   
CLEAN means neither rollout violates. Scenarios rejected before execution are labeled GATED.

For scenarios with a measurable conflict gap, we also report

$$
\Delta g = g ^ { \mathrm { t } } - g ^ { \mathrm { b } } ,\tag{28}
$$

where $g ^ { \mathrm { t } }$ and $g ^ { \mathrm { b } }$ are the minimum conflict gaps under non-degraded and degraded V2X information, respectively.   
Positive $\Delta g$ indicates a loss of physical separation caused by degraded information.

The estimated-state level measures how strongly the vehicle’s perceived state deviates from the physical state. We report hazard slack, decision-epoch Age of Information (AoI), and the type and duration of each estimate–truth divergence. The search level measures refinement efficiency. We report the number of admitted failures, attribution rate, feasibility-gate rejection rate, executions required to reach the first admission, and minimal-witness size after delta debugging. The realization level verifies that the generated scene was executed as intended. Parse success, spawn audits, and route completion are used to detect realization artifacts, while the 100-scene normal-traffic benchmark serves as a negative control for false admissions.

## 3.2 Scenario Generation at Scale

The first experiment evaluates whether plain traffic language can drive the full pipeline at scale. Figure 2 summarizes the 100-scene normal-traffic benchmark. The prompts describe only the traffic scene and environment: they do not name the exercised V2X application, the CARLA town, or any communication fault. The application is inferred by the parser, feasible towns are identified through retrieval, and the benchmark harness assigns the least-used feasible town. As a result, coverage is measured rather than hand-designed. The benchmark covers six V2X applications across eight CARLA towns, with various weather conditions, times of day, vehicle classes, and traffic densities ranging from a single ego vehicle to eleven agents.

Generation is both reliable and faithful. All 100 scenes compile into valid and feasible specifications within at most two parse attempts, with an average of 1.58 attempts per scene, and every scene is grounded to a feasible movement unit and executed to completion. The only recurring compile issue is an out-of-vocabulary maneuver field, which is corrected on retry. Realization quality is also high: 365 of 367 vehicle spawns are clean, all 2048 checked lane markings match the authored OpenDRIVE map, and 253 of 254 vehicles remain on route.

![](images/7617baaf791e769c7f3ad52dcec65616be9ec4b7f6f155dc08dc24c407367d87.jpg)

## a Corridor Prompt

c CARLA Run, Retrieved Corridor  
![](images/11937cb1728656140a0f690f2e564deb069f8a974de885fd05d3ddcdf032b1b8.jpg)  
Figure 4: Corridor traversal along an RSU chain. a, Shared prompt; C is the condition-specific clause. b, Clause, parsed value, and 10-seed results per condition, relative to No V2X. c, Camera frames on the retrieved corridor. d, Junctions with known SPaT. e,f, Ego speed without and with traffic.

Most importantly, the normal-traffic corpus yields zero admitted communication-induced failures under healthy communication, confirming that LAST-V2X does not manufacture failures. The generated scenes also remain physically well behaved, with zero collisions, zero out-of-road distance, and sub-0.15 m tracking error in both rollouts. The paired executions further provide a scene-level measure of information impact. Among the 79 scenes with finite conflict gaps in both rollouts, the ground-truth ego maintains more than 0.5 m additional separation in 33 scenes, with a median gain of 8.5 m and a maximum of 112.3 m. Figure 3 shows one representative dense control scene in detail. Overall, these results show that plain traffic language is sufficient to drive LAST-V2X to broad and faithful V2X scenario generation without introducing spurious failures.

## 3.3 Corridor Traversal along an RSU Chain

The corridor study measures what infrastructure ahead of the trip buys: a connected vehicle’s information horizon need not end at its own radio. Like the main benchmark, it is driven from language (Fig. 4a). One prompt describes the corridor, four signalized junctions with a shared 6/3/12 s plan and offsets timed against a free-flowing driver, and differs between runs only in whether the road is empty or the ego follows eight unequipped cars, and in one information clause (Fig. 4b): no V2X; direct SPaT, where each junction’s RSU broadcasts only its own timing over a 50 m radio; and an RSU relay chain, where backhaul-connected RSUs rebroadcast the whole corridor’s SPaT at 5 ms per hop, so one reachable RSU delivers every junction ahead. All six prompts compile to the intended condition on the first parse; the verifier’s only repairs lengthen a too-short episode. Retrieval binds the corridor to a Town05 arterial with junctions 91, 88 and 54 m apart, and the harness then sweeps the offset plan (aligned, green wave, adverse), the radio range (50, 100, 150 m) and 10 seeds: 420 CARLA runs with no red-light entry and no collision.

b Arterial

“A clear noon on a straight multi-lane arterial. The ego is the last of three cars in one lane, all at 12 m/s: a middle car 30 meters ahead of the ego and a lead car 60 meters ahead. Vehicle radios reach 45 meters, so the ego cannot hear the lead, and the middle car blocks its view of it. Three seconds in, the lead brakes hard to a stop. [clause, in d] Test the emergency electronic brake light …”

![](images/edb02424b8cb90f92f62dffcc8d3e3acf74ac97533849cc753f273ab2cd2da7c.jpg)

![](images/9b645b87f38f990bf248727cda562947298214563b80b2be1b455ed8f8cde4c1.jpg)

“A clear noon at an unsignalized four-way junction whose signals are dark. The ego goes straight at 10 m/s from 120 meters out; a car approaches from the ego's right at 10 m/s from 112 meters out, timed to reach the crossing point slightly before the ego. Vehicle radios send BSMs at 10 Hz and reach 45 meters. [clause, in h] Test intersection movement assist …”

![](images/d1521ac040d5f118f9b0126142ca5a39a08e61ec3c925c543be4026128945998.jpg)

![](images/63a80d095ce5d8fa750b37e23a086be69eedee442488dcd962b5618cc80c5281.jpg)

![](images/7f855b2504bc55b9b6c6a7de3be3dc6d5c23eef7e86d27be9debdebb8bb977b6.jpg)

![](images/cad63c9de95cf479fea01e732ab4d5563413c4a52563fac8309d73532fea2fa3.jpg)  
Lead Lead brakes Brake warning V2V relay Collision Frame instant Conflict point BSMs

Figure 5: Multi-CV cooperation: EEBL relay chain (a–d) and IMA mutual yield (e–h). a,e, Prompts. b,f, Retrieved scenes. c, CV3 speed and gap. g, Distance between the crossing vehicles. d,h, Clause, CARLA frame, and 10-seed results per condition.

The chain’s value concentrates where the radio is weakest. At R = 50 m, below the junction spacing, direct coverage is intermittent, and the relay lifts the mean number of junctions known ahead from 0.5 to 2.6 (Fig. 4d). With the horizon comes smoothness. Against adverse offsets on the empty corridor, GLOSA over the chained SPaT cuts stops from 1.9 to 1.0 and the mean acceleration magnitude from 1.70 to $1 . 2 1 \mathrm { m } / \mathrm { { s } ^ { 2 } }$ relative to no V2X, while direct SPaT at the same range changes little (1.9 stops, 1.62 m/s<sup>2</sup>; Fig. 4b); between J1 and J3 the relayed ego holds about 10 m/s where the others brake and re-accelerate (Fig. 4e). Travel time stays signal-bound (47.4 s without V2X versus 46.6 s with the relay chain), and the one stop that remains, at the first light, which every condition meets on red, is slightly longer than the no-V2X ego’s total stopped time (11.6 versus 10.5 s): the corridor’s benefit is fewer stops and gentler control, not higher speed. The chain at R = 50 m matches direct SPaT at three times the range (1.0 stops, 1.23 m/s<sup>2</sup> at 150 m); infrastructure cooperation substitutes for radio range.

With the eight-car platoon ahead, the queue sets the pace and travel time is the same in all three conditions (98 s, averaged over the offset plans). What changes is how that time is spent: the relay chain cuts the ego’s time stopped from 31.2 to 21.5 s and its mean acceleration magnitude from 0.88 to 0.74 m/s<sup>2</sup>, while direct SPaT, whose short radio rarely sees past the queue, changes neither (31.3 s, 0.86 m/s<sup>2</sup>; Fig. 4b,f). The effect is largest under the green-wave plan, where idling falls from 34.0 to 13.3 s: the ego glides toward queues it knows will discharge instead of braking into them. In traffic, corridor information buys smoothness rather than time.

## 3.4 Cooperative Scenarios with Multiple Connected Vehicles

The multi-CV study makes cooperation itself the object under test: with every connected vehicle owning an estimated state store and a decision policy, what two vehicles achieve together becomes measurable (Fig. 5). Both experiments are again stated in language, one prompt per condition, and all six prompts compile to the intended condition on the first parse; retrieval binds the brake-warning chain to a straight Town05 arterial and the crossing to Town05 junction 1722, and each condition runs for 10 seeds. In the EEBL V2V relay chain, a lead vehicle brakes hard beyond the third vehicle’s radio range and behind an occluding second vehicle, so whether CV3 learns of the event in time depends on whether CV2 rebroadcasts the warning vehicle-to-vehicle. With the V2V relay, CV3 brakes 1.49 s after the lead and keeps a 22.7 m minimum gap, with no collision in 10 seeds. Without it, CV3 waits 3.52 s for visual evidence, the gap closes to 0.1 m and 7 of 10 runs end in a rear-end collision; with CV2 unequipped, CV3 brakes after 4.21 s and 9 of 10 runs collide (Fig. 5c,d). The V2V relay hop converts an occluding vehicle into infrastructure worth 2.0 s of warning.

![](images/74f46c0c7bcabe9f7a6aba11a43261067d7fbb9b2a524a175187a7f49ee0f758.jpg)

![](images/b493897e8e6202dfe8d35a2e197238b9273ba95cc3306f9a580ca5fd94b409fd.jpg)  
c Search Closes the Safety Margin

![](images/06ba1fdff13e31b8eccdf36a8909327b3d4cfb8f8ef58ee206a0a2f44daf5125.jpg)  
Figure 6: Scenario refinement on the 50-scene fault suite. a, Outcome by fault class. b, Executions or prompt revisions to first admission. c, Median best margin during search (shaded: IQR).

The IMA mutual-yield experiment measures the other face of cooperation. Two crossing vehicles timed to conflict interleave cleanly when both are equipped: a 12.7 m closest approach and a 2.04 s post-encroachment time, with the later arrival yielding in 10 of 10 seeds. With one or neither equipped, the encounter degenerates to a 5.6 m near miss with a 0.28 s post-encroachment time and contact in 2 of 10 seeds, identically in both cases (Fig. 5g,h). Cooperation is pairwise: one-sided equipage buys nothing when the counterpart is silent. This restates the corpus’s silent-miss finding at the scale of two deciding vehicles.

## 3.5 Scenario Refinement at Scale

The refinement experiment evaluates whether ineffective V2X tests can be systematically improved through specification space search and LLM self-reflection. The fault suite is expanded from 12 to 50 prompts spanning six applications and diverse communication faults, including intentionally weak, mistimed, and ineffective cases. All scenarios are first executed without refinement. The 43 non-admitted cases are then refined using specification-space search with a budget of 120 candidate mutants per scene and LLM self-reflection with up to five revisions. To measure each mechanism’s contribution, the two loops of Algorithm 1 are applied separately to every target: search mutates the compiled scenario, and reflection revises the prompt, with each revision compiled and executed once.

The initial execution admits only 7 of the 50 scenarios. Most missing, delayed, or stale messages do not produce failures because the application can detect or tolerate them, whereas incorrect but apparently valid information is more likely to alter vehicle behavior. This result reinforces an important point: injecting a communication fault does not by itself create a meaningful V2X safety test.

Across the 50 scenarios, refinement increases admitted failures from 7 to 32 (Fig. 6a). Specification-space search finds failures in 23 of the 43 refinement targets, with a median of 8 executions to the first admission and no successful case requiring more than 27 executions. LLM self-reflection rescues 8 targets, typically within one revised prompt (Fig. 6b). Six of these are also reachable through parameter search, while two require semantic changes beyond the bounded search space. The two refinement mechanisms are therefore complementary: search adjusts quantitative conditions such as fault timing and interaction severity, whereas self-reflection repairs semantically or structurally ineffective test intents.

For scenarios successfully converted by search, the best safety margin decreases to a median of 7% of its initial value, compared with 80% for scenarios that remain fail-safe (Fig. 6c). Delta debugging further reduces each admitted case to a 1-minimal witness with a single fault. Of the 18 scenarios that remain non-admitted, most either fail under both communication conditions and are therefore non-attributable, or remain fail-safe despite refinement. These results show that LAST-V2X can systematically improve ineffective tests without forcing every scenario into an unsafe outcome.

Figure 7 illustrates how self-reflection repairs an ineffective test. In rlvw\_late\_freeze, the original SPaT freeze begins after the ego has already made its stopping decision, so the scenario remains clean. Using the execution evidence, the LLM identifies the mistimed fault and revises the scenario so that the frozen SPaT persists through the critical decision window. After refinement, the message-driven ego enters the intersection on red while the matched non-degraded ego stops, converting the original scenario into a causally verified communication-induced failure.

## a Prompt Revision by LLM Self-Reflection

![](images/3e6176f56d33f49c0b61cb0d1c5a94061fc5fed59600bd97e9e0666974098c59.jpg)  
Figure 7: Refinement example rlvw\_late\_freeze. a, Prompt revisions by LLM self-reflection. b,c, Paired runs of the original and revised prompts: ego and ground-truth twin, true and received signal phases, and CARLA frames.

## 3.6 Ablation and Sensitivity Studies

Table 2 evaluates the contribution of each major pipeline component by removing it individually and comparing the resulting performance with the full system. The ablations test launch conditioning, height-aware blockage, the ground-truth rollout, the feasibility gate, the scene planner, and prompt reflection.

Several findings are notable. First, the ground-truth rollout is essential for causal attribution. Among the 112 live scenes, 60 contain at least one decision made under a divergent estimated state. A divergence-only monitor would therefore raise 60 alarms, whereas the paired rollout identifies only one communication-induced failure, avoiding widespread false attribution. Second, the channel sensitivity study shows that building blockage is the dominant communication parameter. On Town10, reducing blockage loss from 5 to 1 dB/m increases in-range road coverage from 47.7% to

Table 2: Ablation of LAST-V2X components. Each block removes one component from the full pipeline; ↑/↓ marks whether a higher or lower value is better, and the better value is in bold.
<table><tr><td rowspan=1 colspan=1>Component and metric</td><td rowspan=1 colspan=1>With</td><td rowspan=1 colspan=1>Without</td><td rowspan=1 colspan=1>Component and metric</td><td rowspan=1 colspan=1>With</td><td rowspan=1 colspan=1>Without</td></tr><tr><td rowspan=1 colspan=1>Ground-truth rollout</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Height-aware blockage</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>Alarms raised (112 live scenes)False alarms among them              ↓</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>Divergences (15 healthy scenes)        Y</td><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=1>60</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>59</td><td rowspan=1 colspan=1>In-range coverage, Town10 (%)        个</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>47.7</td></tr><tr><td rowspan=1 colspan=1>Feasibility gate</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Scene planner (multi-approach junctions)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>IMA admissions per 150 executions    ↑Seeds with zero admissions (of 3)</td><td rowspan=2 colspan=1>29.00</td><td rowspan=1 colspan=1>15.3</td><td rowspan=1 colspan=1>Conflict gap, ground-truth ego (m)</td><td rowspan=1 colspan=1>10.8</td><td rowspan=1 colspan=1>49.8</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Conflict gap, message-driven ego (m)</td><td rowspan=1 colspan=1>2.4</td><td rowspan=1 colspan=1>5.7</td></tr><tr><td rowspan=1 colspan=1>Launch conditioning</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Prompt reflection</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=1>Opening cross-track error, median (m)Opening cross-track error, worst (m)Spurious ground-truth collisions</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>1.22</td><td rowspan=1 colspan=1>Non-admitted prompts rescued (of 43) 个</td><td rowspan=2 colspan=1>82</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>2.26</td><td rowspan=1 colspan=1>6.63</td><td rowspan=1 colspan=1>Scenes admitted beyond search reach  个</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

95.8%, while changing transmit power by ±10 dB shifts coverage by less than four percentage points. Height-aware blockage also avoids the systematic over-attenuation produced by a height-blind model. Third, the feasibility gate and scene planner matter primarily when their corresponding failure modes are present. Without the feasibility gate, infeasible mutants enter the search and reduce IMA admissions from 29.0 to 15.3 per 150-execution ablation budget, averaged over three seeds, while RLVW is largely unaffected because its mutations remain feasible. Similarly, the scene planner has little effect on simple junctions but becomes critical at multi-approach intersections, where heuristic grounding can eliminate the intended conflict. Finally, prompt reflection extends the refinement space beyond bounded parameter search. It rescues eight non-admitted prompts, including two cases that search alone cannot reach. When refinement is unsuccessful, reflection does not force an admission; instead, it identifies why the scenario remains ineffective or non-attributable.

## 4 Conclusion

This paper presented LAST-V2X, a language-driven, adaptive framework for automated scenario synthesis and safety testing of vehicle-to-everything (V2X) applications. By integrating large language models with domain knowledge, physical traffic simulation, and communication modeling, LAST-V2X transforms natural-language test descriptions into executable scenarios that capture the interactions between vehicle behavior and V2X information. Unlike conventional scenario-generation approaches that primarily focus on physical traffic conditions, LAST-V2X explicitly examines how communication degradation propagates through application decisions to affect driving safety. A paired execution mechanism distinguishes communication-induced failures from unsafe traffic conditions, while execution-guided parameter search and LLM self-reflection adaptively refine ineffective scenarios and identify minimal reproducible failures.

Experiments across diverse CARLA environments and V2X safety applications demonstrated the effectiveness of the proposed framework. LAST-V2X reliably generated executable scenarios from natural-language descriptions while avoiding spurious failure attribution under healthy communication. The paired execution mechanism further demonstrated the importance of causal verification, as discrepancies between received information and physical ground truth do not necessarily lead to unsafe outcomes. Adaptive refinement substantially improved the discovery of communication-induced failures by correcting ineffective fault configurations and revising test descriptions based on simulation evidence. Additional experiments highlighted how communication coverage, information availability, and cooperation among connected vehicles influence safety and driving behavior. Collectively, these findings demonstrate the value of integrating language-driven scenario generation, causal verification, and adaptive refinement into a unified V2X testing framework.

Several limitations remain. First, the communication model and fault configurations rely on representative parameters rather than comprehensive measurements from real-world deployments. Second, the evaluated safety applications employ simplified, deterministic decision logic that may not fully capture the complexity of production V2X systems. Third, although the main failure mechanisms were validated in CARLA, large-scale adaptive search was conducted using a simplified backend, and its effectiveness under fully integrated simulation remains to be established. Finally, the current framework assumes fixed-time traffic signals and does not explicitly distinguish communication errors from changes in signal timing caused by adaptive traffic control.

Future research will focus on improving the realism, scalability, and practical applicability of LAST-V2X. Integrating field-measured communication characteristics and more advanced V2X application logic will enable more representative evaluations of real-world connected vehicle systems. Extending adaptive refinement to fully integrated CARLA simulations and more complex traffic environments will further assess the framework’s generalizability. Additional directions include supporting actuated signal control, expanding the range of communication and behavioral uncertainties, and using generated failure evidence to inform communication infrastructure design and V2X safety improvements. Ultimately, LAST-V2X provides a foundation for moving beyond manually specified test cases toward automated, evidence-driven safety evaluation of connected vehicle systems.

## References

[1] Ran Yi, Yifan Yao, Fan Pu, Yang Zhou, and Xin Wang. Cooperative CAV mandatory lane-change control enabled by V2I. Communications in Transportation Research, 4:100126, 2024. doi: 10.1016/j.commtr.2024.100126.

[2] Fengze Yang, Bo Yu, Yang Zhou, Xuewen Luo, Zhengzhong Tu, and Chenxi Liu. Edge-based multimodal sensor data fusion with vision-language-action (VLA) model for real-time autonomous vehicle accident avoidance. Engineering Applications of Artificial Intelligence, 180:115186, 2026. doi: 10.1016/j.engappai.2026.115186.

[3] Hangyu Li, Bofeng Cao, Zhaohui Liang, Wuzhen Li, Juyoung Oh, Yuxuan Chen, Shixiao Liang, Hang Zhou, Chengyuan Ma, Jiaxi Liu, et al. CATS-V2V: A real-world vehicle-to-vehicle cooperative perception dataset with complex adverse traffic scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pages 2294–2303, 2026. doi: 10.1109/CVPRF71909.2026.00224.

[4] Xiangbo Gao, Yuheng Wu, Fengze Yang, Xuewen Luo, Keshu Wu, Xinghao Chen, Yuping Wang, Chenxi Liu, Yang Zhou, and Zhengzhong Tu. AirV2X: Unified air-ground vehicle-to-everything collaboration. arXiv preprint arXiv:2506.19283, 2025.

[5] Haibao Yu, Wenxian Yang, Jiaru Zhong, Zhenwei Yang, Siqi Fan, Ping Luo, and Zaiqing Nie. End-to-end autonomous driving through V2X cooperation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 9598–9606, 2025. doi: 10.1609/aaai.v39i9.33040.

[6] SAE International. V2X communications message set dictionary. SAE Standard J2735\_202409, September 2024.

[7] Keshu Wu, Yang Cheng, Steven T. Parker, Bin Ran, and David A. Noyce. Development of the data pipeline for a connected vehicle corridor. In International Conference on Transportation and Development 2023, pages 218–230. ASCE, 2023. doi: 10.1061/9780784484876.020.

[8] Haibao Yu, Wenxian Yang, Hongzhi Ruan, Zhenwei Yang, Yingjuan Tang, Xu Gao, Xin Hao, Yifeng Shi, Yifeng Pan, Ning Sun, et al. V2X-Seq: A large-scale sequential dataset for vehicle-infrastructure cooperative perception and forecasting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5486–5495, 2023.

[9] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. OpenAI GPT-5 system card. arXiv preprint arXiv:2601.03267, 2025.

[10] Yuping Wang, Shuo Xing, Can Cui, Renjie Li, Hongyuan Hua, Kexin Tian, Zhaobin Mo, Xiangbo Gao, Marco Pavone, Yang Zhou, Jiachen Li, and Zhengzhong Tu. Generative AI for autonomous driving: Frontiers and opportunities. ACM Computing Surveys, 2026. doi: 10.1145/3838726.

[11] Keshu Wu, Pei Li, Yang Zhou, Rui Gan, Junwei You, Yang Cheng, Jingwen Zhu, Steven T. Parker, Bin Ran, David A. Noyce, and Zhengzhong Tu. V2X-LLM: Enhancing V2X integration and understanding in connected vehicle corridors. arXiv preprint arXiv:2503.02239, 2025.

[12] Junwei You, Zhuoyu Jiang, Zilin Huang, Haotian Shi, Rui Gan, Keshu Wu, Xi Cheng, Xiaopeng Li, and Bin Ran. V2X-VLM: End-to-end V2X cooperative autonomous driving through large vision-language models. Transportation Research Part C: Emerging Technologies, 183:105457, 2026. doi: 10.1016/j.trc.2025.105457.

[13] Changxing Liu, Genjia Liu, Zijun Wang, Jinchang Yang, and Siheng Chen. CoLMDriver: LLM-based negotiation benefits cooperative autonomous driving. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 25951–25960, 2025. doi: 10.1109/ICCV51701.2025.02407.

[14] Hsu-kuang Chiu, Ryo Hachiuma, Chien-Yi Wang, Stephen F. Smith, Yu-Chiang Frank Wang, and Min-Hung Chen. V2V-LLM: Vehicle-to-vehicle cooperative autonomous driving with multimodal large language models. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 12538–12545, 2026. doi: 10.1109/ICRA57385.2026.11696625.

[15] Xiangbo Gao, Yuheng Wu, Rujia Wang, Chenxi Liu, Yang Zhou, and Zhengzhong Tu. LangCoop: Collaborative driving with language. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 4226–4237, 2025. doi: 10.1109/CVPRW67362.2025.00406.

[16] Bo-Kai Ruan, Hao-Tang Tsui, Yung-Hui Li, and Hong-Han Shuai. Traffic scene generation from natural language description for autonomous vehicles with large language model. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 935–944, 2026.

[17] Jiawei Zhang, Chejian Xu, and Bo Li. ChatScene: Knowledge-enabled safety-critical scenario generation for autonomous vehicles. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15459–15469, 2024.

[18] Daniel J. Fremont, Edward Kim, Tommaso Dreossi, Shromona Ghosh, Xiangyu Yue, Alberto L. Sangiovanni-Vincentelli, and Sanjit A. Seshia. Scenic: A language for scenario specification and data generation. Machine Learning, 112(10):3805–3849, 2023. doi: 10.1007/s10994-021-06120-5.

[19] Wenhao Ding, Yulong Cao, Ding Zhao, Chaowei Xiao, and Marco Pavone. RealGen: Retrieval augmented generation for controllable traffic scenarios. In Computer Vision – ECCV 2024, Lecture Notes in Computer Science, pages 93–110. Springer, 2024. doi: 10.1007/978-3-031-73033-7\_6.

[20] Zihao Sheng, Zilin Huang, Yansong Qu, Yue Leng, and Sikai Chen. Talk2Traffic: Interactive and editable traffic scenario generation for autonomous driving with multimodal large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 3788–3797, 2025. doi: 10.1109/CVPRW67362.2025.00364.

[21] Zhi Tu, Liangkun Niu, Wei Fan, and Tianyi Zhang. Multi-modal traffic scenario generation for autonomous driving system testing. Proceedings of the ACM on Software Engineering, 2(FSE):1733–1756, 2025. doi: 10.1145/3729348.

[22] Xuan Cai, Xuesong Bai, Zhiyong Cui, Danmu Xie, Daocheng Fu, Haiyang Yu, and Yilong Ren. Text2Scenario: Text-driven scenario generation for autonomous driving test. Automotive Innovation, 9(1):102–127, 2026. doi: 10.1007/s42154-025-00374-8.

[23] Keshu Wu, Hao Zhang, Rui Gan, Xiangbo Gao, Xiaopeng Li, Zhengzhong Tu, and Yang Zhou. Aurora: A natural language-driven agentic framework for understanding, reasoning, and orchestrating reliable air-ground co-simulation. arXiv preprint arXiv:2609.19527, 2026.

[24] Hao Gao, Jingyue Wang, Wenyang Fang, Jingwei Xu, Yunpeng Huang, Taolue Chen, and Xiaoxing Ma. LASER: Script execution by autonomous agents for on-demand traffic simulation. In Proceedings ofthe 16th International Conference on Internetware, pages 84–95. ACM, 2025. doi: 10.1145/3755881.3755905.

[25] Shuhan Tan, Boris Ivanovic, Yuxiao Chen, Boyi Li, Xinshuo Weng, Yulong Cao, Philipp Kraehenbuehl, and Marco Pavone. Promptable closed-loop traffic simulation. In Proceedings ofThe 8th Conference on Robot Learning, volume 270 of Proceedings ofMachine Learning Research, pages 5087–5105. PMLR, 2025.

[26] Karim Elmaaroufi, Devan Shanker, Ana Cismaru, Marcell Vazquez-Chanlatte, Alberto L. Sangiovanni-Vincentelli, Matei Zaharia, and Sanjit A. Seshia. ScenicNL: Generating probabilistic scenario programs from natural language. In Conference on Language Modeling (COLM), 2024.

[27] Yongqi Zhao, Ji Zhou, Dong Bi, Tomislav Mihalj, Jia Hu, and Arno Eichberger. A survey on the application of large language models in scenario-based testing of automated driving systems. IEEE Transactions on Intelligent Transportation Systems, 27(5):5001–5023, 2026. doi: 10.1109/TITS.2026.3651004.

[28] Pei Li, Keshu Wu, Yang Cheng, Steven T. Parker, and David A. Noyce. How does C-V2X perform in urban environments? Results from real-world experiments on urban arterials. IEEE Transactions on Intelligent Vehicles, 9(1):2520–2530, 2024. doi: 10.1109/TIV.2023.3326735.

[29] Keshu Wu, Pei Li, Yang Cheng, Steven T. Parker, Bin Ran, David A. Noyce, and Xinyue Ye. A digital twin framework for physical-virtual integration in V2X-enabled connected vehicle corridors. IEEE Transactions on Intelligent Transportation Systems, 26(9):14407–14420, 2025. doi: 10.1109/TITS.2025.3564351.

[30] Xiwen Liao, Supeng Leng, Yao Sun, Ke Zhang, and Muhammad Ali Imran. A digital-twin-based traffic guidance scheme for autonomous driving. IEEE Internet of Things Journal, 11(22):36829–36840, 2024. doi: 10.1109/JIOT.2024.3429534.

[31] Hao Zhang, Ximin Yue, Kexin Tian, Sixu Li, Keshu Wu, Zihao Li, Dominique Lord, and Yang Zhou. Virtual roads, smarter safety: A digital twin framework for mixed autonomous traffic safety analysis. IEEE Internet of Things Journal, 13(4):6045–6058, 2026. doi: 10.1109/JIOT.2025.3637688.

[32] Qingzhao Zhang, Shuowei Jin, Ruiyang Zhu, Jiachen Sun, Xumiao Zhang, Qi Alfred Chen, and Z. Morley Mao. On data fabrication in collaborative vehicular perception: Attacks and countermeasures. In Proceedings ofthe 33rd USENIX Security Symposium (USENIX Security), pages 6309–6326. USENIX Association, 2024.

[33] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. CARLA: An open urban driving simulator. In Proceedings ofthe 1st Annual Conference on Robot Learning (CoRL), volume 78 of Proceedings ofMachine Learning Research, pages 1–16. PMLR, 2017.

[34] Quanyi Li, Zhenghao Peng, Lan Feng, Qihang Zhang, Zhenghai Xue, and Bolei Zhou. MetaDrive: Composing diverse driving scenarios for generalizable reinforcement learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(3):3461–3475, 2023. doi: 10.1109/TPAMI.2022.3190471.

[35] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng, and Henry X. Liu. Intelligent driving intelligence test for autonomous vehicles with naturalistic and adversarial environment. Nature Communications, 12(1):748, 2021. doi: 10.1038/s41467-021-21007-8.

[36] Shuo Feng, Haowei Sun, Xintao Yan, Haojie Zhu, Zhengxia Zou, Shengyin Shen, and Henry X. Liu. Dense reinforcement learning for safety validation of autonomous vehicles. Nature, 615(7953):620–627, 2023. doi: 10.1038/s41586-023-05732-2.

[37] Keshu Wu, Zihao Li, Sixu Li, Xinyue Ye, Dominique Lord, and Yang Zhou. AI2-Active Safety: AI-enabled interaction-aware active safety analysis with vehicle dynamics. arXiv preprint arXiv:2505.00322, 2025.

[38] Keshu Wu, Yang Zhou, Haotian Shi, Dominique Lord, Bin Ran, and Xinyue Ye. Hypergraph-based motion generation with multi-modal interaction relational reasoning. Transportation Research Part C: Emerging Technologies, 180:105349, 2025. doi: 10.1016/j.trc.2025.105349.

[39] Wenhao Ding, Baiming Chen, Minjun Xu, and Ding Zhao. Learning to collide: An adaptive safety-critical scenarios generating method. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 2243–2250, 2020. doi: 10.1109/IROS45743.2020.9340696.

[40] Jingkang Wang, Ava Pun, James Tu, Sivabalan Manivasagam, Abbas Sadat, Sergio Casas, Mengye Ren, and Raquel Urtasun. Advsim: Generating safety-critical scenarios for self-driving vehicles. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9909–9918, 2021.

[41] Linrui Zhang, Zhenghao Peng, Quanyi Li, and Bolei Zhou. CAT: Closed-loop adversarial training for safe end-to-end driving. In Proceedings ofthe 7th Conference on Robot Learning (CoRL), volume 229 of Proceeding ofMachine Learning Research, pages 2357–2372. PMLR, 2023.

[42] Chejian Xu, Wenhao Ding, Weijie Lyu, Zuxin Liu, Shuai Wang, Yihan He, Hanjiang Hu, Ding Zhao, and Bo Li. SafeBench: A benchmarking platform for safety evaluation of autonomous vehicles. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, volume 35, pages 25667–25682, 2022. doi: 10.52202/068431-1861.

[43] Shuyang Li, Meng Ma, Talha Azfar, and Ruimin Ke. ChatSUMO agent: An LLM-based agent for conversational traffic simulation in SUMO. Transportation Research Part C: Emerging Technologies, 190:105759, 2026. doi: 10.1016/j.trc.2026.105759.

[44] Shuncheng Tang, Zhenya Zhang, Jixiang Zhou, Lei Lei, Yuan Zhou, and Yinxing Xue. LeGEND: A top-down approach to scenario generation of autonomous driving systems assisted by large language models. In Proceedings ofthe 39th IEEE/ACM International Conference on Automated Software Engineering (ASE), pages 1497–1508. ACM, 2024. doi: 10.1145/3691620.3695520.

[45] Mingxing Peng, Yuting Xie, Xusen Guo, Ruoyu Yao, Hai Yang, and Jun Ma. LD-Scene: LLM-guided diffusion for controllable generation of adversarial safety-critical driving scenarios. Transportation Research Part C: Emerging Technologies, 188:105694, 2026. doi: 10.1016/j.trc.2026.105694.

[46] Yuewen Mei, Tong Nie, Jian Sun, and Ye Tian. LLM-Attacker: Enhancing closed-loop adversarial scenario generation for autonomous driving with large language models. IEEE Transactions on Intelligent Transportation Systems, 26(10):15068–15076, 2025. doi: 10.1109/TITS.2025.3578383.

[47] Yuewen Mei, Tong Nie, Jian Sun, and Ye Tian. Seeking to collide: Online safety-critical scenario generation for autonomous driving with retrieval augmented large language models. In 2025 IEEE 28th International Conference on Intelligent Transportation Systems (ITSC), pages 943–950, 2025. doi: 10.1109/ITSC60802.2025.11423790.

[48] Yuan Gao, Mattia Piccinini, Korbinian Moller, Amr Alanwar, and Johannes Betz. From words to collisions: LLM guided evaluation and adversarial generation of safety-critical driving scenarios. In 2025 IEEE 28th International Conference on Intelligent Transportation Systems (ITSC), pages 2134–2141, 2025. doi: 10.1109/ITSC60802.2025. 11423486.

[49] Yuhang Wang, Heye Huang, Zhenhua Xu, Kailai Sun, Baoshen Guo, and Jinhua Zhao. Learning from risk: LLM-guided generation of safety-critical scenarios with prior knowledge. Transportation Research Part E: Logistics and Transportation Review, 215:105110, 2026. doi: 10.1016/j.tre.2026.105110.

[50] Hongyi Zhao, Shuo Wang, Qijie He, and Ziyuan Pu. SG-CADVLM: A context-aware decoding powered visionlanguage model for safety-critical scenario generation in autonomous driving. Transportmetrica A: Transport Science, pages 1–27, 2026. doi: 10.1080/23249935.2026.2666104.

[51] Yunsong Zhou, Naisheng Ye, William Ljungbergh, Tianyu Li, Jiazhi Yang, Zetong Yang, Hongzi Zhu, Christoffer Petersson, and Hongyang Li. Decoupled diffusion sparks adaptive scene generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 27760–27770, 2025. doi: 10.1109/ ICCV51701.2025.02577.

[52] An Guo, Yuan Zhou, Haoxiang Tian, Chunrong Fang, Yunjian Sun, Weisong Sun, Xinyu Gao, Anh Tuan Luu, Yang Liu, and Zhenyu Chen. SoVAR: Build generalizable scenarios from accident reports for autonomous driving testing. In Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering (ASE), pages 268–280, 2024. doi: 10.1145/3691620.3695037.

[53] Guoqiang Cai, Bo Fan, Yiwei Dong, Tongfei Li, Yuan Wu, and Yan Zhang. Task-efficiency oriented V2X communications: Digital twin meets mobile edge computing. IEEE Wireless Communications, 31(2):149–155, 2024. doi: 10.1109/MWC.012.2200465.

[54] Stefan Zelenbaba, Benjamin Rainer, Markus Hofer, and Thomas Zemen. Wireless digital twin for assessing the reliability of vehicular communication links. In 2022 IEEE Globecom Workshops (GC Wkshps), pages 1034–1039, 2022. doi: 10.1109/GCWkshps56602.2022.10008559.

[55] Qian Liu, Xiaoqing Qi, Shanshan Liu, Xiuju Cheng, Xin Ke, and Feng Wang. Application of lightweight digital twin system in intelligent transportation. IEEE Journal ofRadio Frequency Identification, 6:729–732, 2022. doi: 10.1109/JRFID.2022.3212169.

[56] Vasilis Maglogiannis, Dries Naudts, Seilendria Hadiwardoyo, Daniel van den Akker, Johann Marquez-Barja, and Ingrid Moerman. Experimental V2X evaluation for C-V2X and ITS-G5 technologies in a real-life highway environment. IEEE Transactions on Network and Service Management, 19(2):1521–1538, 2022. doi: 10.1109/ TNSM.2021.3129348.

[57] Runsheng Xu, Hao Xiang, Xu Han, Xin Xia, Zonglin Meng, Chia-Ju Chen, Camila Correa-Jullian, and Jiaqi Ma. The OpenCDA open-source ecosystem for cooperative driving automation research. IEEE Transactions on Intelligent Vehicles, 8(4):2698–2711, 2023. doi: 10.1109/TIV.2023.3244948.

[58] Daniel Grimm, Marc Schindewolf, David Kraus, and Eric Sax. Co-simulate no more: The CARLA V2X sensor. In 2024 IEEE Intelligent Vehicles Symposium (IV), pages 2429–2436, 2024. doi: 10.1109/IV55156.2024.10588488.

[59] An Guo, Xinyu Gao, Zhenyu Chen, Yuan Xiao, Jiakai Liu, Xiuting Ge, Weisong Sun, and Chunrong Fang. CooTest: An automated testing approach for V2X communication systems. In Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA), pages 1453–1465. ACM, 2024. doi: 10.1145/3650212.3680373.

[60] Sanjit Kaul, Roy Yates, and Marco Gruteser. Real-time status: How often should one update? In Proceedings of IEEE INFOCOM, pages 2731–2735, 2012. doi: 10.1109/INFCOM.2012.6195689.

[61] Keshu Wu, Chenchen Kuai, Zihao Li, Jiwan Jiang, Shiyu Shen, Shian Wang, Chan-Wei Hu, Zhengzhong Tu, and Yang Zhou. Knowledge is not static: Order-aware hypergraph RAG for language models. arXiv preprint arXiv:2604.12185, 2026.

[62] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, pages 8634–8652, 2023. doi: 10.52202/075280-0377.

[63] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, pages 46534–46594, 2023. doi: 10.52202/075280-2019.

[64] OpenAI. GPT-4o system card. arXiv preprint arXiv:2410.21276, 2024.

[65] SAE International. On-board system requirements for V2V safety communications. SAE Standard J2945/1\_202004, April 2020.

## Appendix A Verifier Check Families

The analytic verifier of Module I (Section 2.2) screens every candidate specification before any simulator time is spent. Each check is decidable from the specification and the closed-form AoI relations alone, and returns a finding that names the check, its severity, the specification fields responsible, and a minimal repair, such as the breaching drop count k<sup>∗</sup> of Eq. (16) or a minimum episode duration. Severity determines how a finding is used:

• Error findings are blocking. During intent parsing they are returned to the LLM parser, which revises the specification; during specification-space search (Section 2.5) a mutant with any error finding is counted as gated and is never executed.

• Warning findings do not block execution but mark a scenario whose outcome is unlikely to be attributable or informative, for example because the pristine channel already exceeds an age budget.

• Note findings record a property of the scenario, such as a deliberate silent-miss probe in which every conflict agent is unequipped.

Verifier diagnostics also enter the evidence digest used by prompt-space reflection. Table 3 lists the eleven check families. The conflict-timing check uses an arrival-overlap window of 8 s, and the penetration check warns below a market penetration rate of 0.2.

Table 3: Verifier check families, grouped by severity.
<table><tr><td>Check</td><td>Condition rejected or flagged</td></tr><tr><td colspan="2">error blocking; returned to the parser for repair</td></tr><tr><td>no_effective_fault</td><td>No fault can breach any application budget (a vacuous intent); suggestions name  $k ^ { * }$  or a freeze/skew fault</td></tr><tr><td>anchor_beyond_horizon</td><td>The k-th anchored occurrence, or the ego decision point, falls after the horizon T</td></tr><tr><td>agent_unreachable</td><td>The conflict cannot form</td></tr><tr><td>no_locus</td><td>Retrieval returns no locus; the message names the first filter that empties the candidate set</td></tr><tr><td colspan="2">warn executed, but flagged as unlikely to be attributable</td></tr><tr><td>fault_below_budget</td><td>A particular fault is invisible to the safety criterion</td></tr><tr><td>baseline_blows_budget conflict_timing_mismatch</td><td>The pristine channel already exceeds a budget, so failures are not attributable to faults</td></tr><tr><td></td><td>The arrival gap exceeds the overlap window</td></tr><tr><td>no_closing_speed</td><td>The FCW/EEBL lead vehicle is not closing</td></tr><tr><td>mpr_low_unpinned</td><td>Regimes mix under low market penetration</td></tr><tr><td>noterecorded with the scenario</td><td></td></tr><tr><td>silent_miss_mode</td><td>All conflict agents are pinned unequipped (structural probe)</td></tr><tr><td>outside_rsu_range_at_spawn</td><td>Reception begins mid-approach</td></tr></table>

## Appendix B Channel Model Details

Every V2I and V2V message passes independently through the spatial channel of Module III (Section 2.4); Table 4 lists its default parameters. For a transmitter–receiver pair at distance $d ,$ the received signal strength of Eq. (24) is evaluated as

$$
\mathrm { R S S I } = \operatorname* { m a x } \Bigl ( P _ { \mathrm { t x } } - 1 0 n \log _ { 1 0 } \operatorname* { m a x } ( d , 1 \mathrm { m } ) - B _ { \mathrm { b l k } } , \ P _ { \mathrm { f l o o r } } \Bigr ) ,\tag{29}
$$

and a link with $d > R$ is assigned the noise floor.

Height-Aware Blockage The blockage term of Eq. (24) is

$$
{ \cal B } _ { \mathrm { b l k } } \ = \ \operatorname* { m i n } \Bigl ( \eta \int _ { \mathrm { p a t h } } \operatorname* { m i n } \bigl ( 1 , \ : h _ { \mathrm { e x c } } ( s ) / h _ { \mathrm { s a t } } \bigr ) d s , { \cal B } _ { \mathrm { N L O S } } \Bigr ) ,\tag{30}
$$

where $h _ { \mathrm { e x c } } ( s ) = \operatorname* { m a x } \left( 0 , h _ { \mathrm { b } } ( s ) - h _ { \mathrm { r a y } } ( s ) \right)$ is the height by which the building at path position s rises above the sight line, and $h _ { \mathrm { r a y } } ( s )$ falls linearly from the RSU antenna height $h _ { \mathrm { R S U } }$ to the receiver antenna height $h _ { \mathrm { O B U } }$ . The integral is evaluated by sampling the ray at steps of $\Delta s .$ . Loss therefore accrues only where a building stands above the line of sight, grows with how far above it stands until it saturates at $h _ { \mathrm { s a t } }$ , and is capped at $B _ { \mathrm { N L O S } } \mathrm { . }$ however deep the obstruction, energy diffracted over rooflines and reflected from facades still reaches the receiver.

Latency and Delivery Latency grows linearly once the signal falls below a knee,

$$
L = L _ { 0 } + \kappa \operatorname* { m a x } \bigl ( 0 , \mathrm { R S S I _ { k n e e } } - \mathrm { R S S I } \bigr ) ,\tag{31}
$$

and per-message delivery is logistic in RSSI,

$$
p _ { \mathrm { d e l } } = \frac { 1 } { 1 + \mathrm { e x p } \big ( ( \mathrm { R S S I } _ { 5 0 } - \mathrm { R S S I } ) / \nu \big ) } .\tag{32}
$$

Together, Eqs. (31) and (32) realize the mapping g of Eq. (24). A message is lost if its RSSI is at the noise floor or a Bernoulli draw with probability $1 - p _ { \mathrm { d e l } }$ fails; otherwise it is delivered after L. The semantic faults of Module III act on top of this channel.

Table 4: Default channel parameters. <sup>†</sup>The corridor study varies R over 50, 100, and 150 m.
<table><tr><td>Parameter</td><td>Symbol</td><td>Value</td><td>Parameter</td><td>Symbol</td><td>Value</td></tr><tr><td>Path loss</td><td></td><td></td><td>Height-aware blockage</td><td></td><td></td></tr><tr><td>Transmit power</td><td> $P _ { \mathrm { t x } }$ </td><td>0dBm</td><td>Building loss rate</td><td>η</td><td>5 dB/m</td></tr><tr><td>Path-loss exponent</td><td>n</td><td>2.0</td><td>Ray sampling step</td><td> $\Delta s$ </td><td>1m</td></tr><tr><td>Noise floor</td><td> $P _ { \mathrm { f l o o r } }$ </td><td>-100 dBm</td><td>RSU antenna height</td><td>hRSU</td><td>6.5m</td></tr><tr><td>Communication radius</td><td>R</td><td>100 m†</td><td>Receiver antenna height</td><td>hOBU</td><td>1.5m</td></tr><tr><td>Latency and delivery</td><td></td><td></td><td>Saturation excess height</td><td> $h _ { \mathrm { s a t } }$ </td><td>10m</td></tr><tr><td>Base latency</td><td> $L _ { 0 }$ </td><td>10ms</td><td>NLOS loss cap</td><td>BNLOS</td><td>25 dB</td></tr><tr><td>Latency slope</td><td>κ</td><td>0.2 ms/dB</td><td>Message rates</td><td></td><td></td></tr><tr><td>Latency knee</td><td>RSSIknee</td><td>-70 dBm</td><td>SPaT / MAP / BSM</td><td> $1 / T _ { s }$ </td><td>10 / 1 / 10 Hz</td></tr><tr><td>Delivery midpoint</td><td>RSSI50</td><td>-90 dBm</td><td></td><td></td><td></td></tr><tr><td>Delivery scale</td><td>ν</td><td>5dB</td><td></td><td></td><td></td></tr></table>

## Appendix C Fault Operators

Each fault is a typed primitive that states what is corrupted (the target stream, SPaT, MAP, or BSM, and the field), when, and how much. Its activation window is $[ t _ { a } + o , t _ { a } + o + \tau )$ , where $t _ { a }$ is the time of the k-th occurrence of an anchor event, o is an offset that may be negative, and an unset duration τ keeps the fault active until the scenario ends. The anchor events are sim\_start, green\_onset, amber\_onset, red\_onset, phase\_change, ego\_at\_decision\_point, and ego\_enters\_range; signal anchors resolve against the bound signal group and kinematic anchors against the messagedriven ego. A fault may be restricted to one sender, and each records the provenance of its magnitude (measured or representative); the experiments in this paper use representative magnitudes. Table 5 defines the seven operators together with the parameter ranges explored by specification-space search. Search additionally moves window offsets in steps of ±2.5 s within $[ - 1 0 , 1 0 ] \mathrm { s } ,$ sets durations to {2, 5, 10} s or open, and changes the anchor event.

A freeze captures whatever the payload shows when its window opens. A feed that should appear stuck at green is therefore anchored before the amber onset (for example, amber\_onset with an offset of −3 s or earlier); a window anchored at red\_onset captures yellow instead, and the scenario typically stops safely.

## Appendix D Safety Application Logic

Each application is a deterministic kinematic-threshold function $\pi _ { a } ( \beta , x ) \ ( \mathrm { E q . } \ ( 2 ) )$ . The same function drives both rollouts: the message-driven ego evaluates it on the reconstructed state $\beta$ and the ground-truth ego on the corresponding state derived from W, so any difference in their outcomes is attributable to the information channel. Braking decisions use a comfortable deceleration $a _ { c } = 3 . 5 \mathrm { m } / \mathrm { s } ^ { 2 }$ and a reaction time $t _ { r } = 1 . 0 \mathrm { s } ,$ , giving the stopping distance

$$
d _ { s } ( v ) = v t _ { r } + \frac { v ^ { 2 } } { 2 a _ { c } } ,\tag{33}
$$

and the decision epoch of a signal application is the last point at which stopping remains comfortable, $d \leq d _ { s } ( v )$ , where d is the distance to the stop bar. Table 6 lists each decision rule and the violation that sets V = 1 in Eq. (7).

Violations are adjudicated identically for both rollouts. A red entry is the crossing of the stop bar while the true signal of the ego’s group is red; being beyond the bar when a later red begins is legal. A collision requires a center-to-center distance below the stated threshold while either vehicle moves faster than 0.5 m/s. For FCW and EEBL, only vehicles ahead of the ego count, and the ego must be the striking vehicle. In CARLA, a vehicle counts toward a conflict only if it lies within 2.5 m of the ego’s route, so that adjacent-lane traffic passing close to the ego is not scored as a crash.

Table 5: Fault operators, their target streams, and the parameter ranges explored by specification-space search.
<table><tr><td>Operator</td><td>Target</td><td>Effect</td><td>Search range</td></tr><tr><td colspan="4">Delivery faults</td></tr><tr><td>drop</td><td>any</td><td>Suppresses the next k broadcasts in the window (all if k is unset)</td><td> $k \in [ 1 , 4 0 ]$ </td></tr><tr><td>burst_drop delay</td><td>any</td><td>Gilbert-Elliott burst loss with transitions  $p _ { \mathrm { g b } } , p _ { \mathrm { b g } }$  Constant delay plus jitter, optionally applied after reception (decode</td><td>and a bad-state loss rate pathology library [0.05, 5] s</td></tr><tr><td></td><td>any</td><td>delay)</td><td></td></tr><tr><td colspan="4">Timestamp fault</td></tr><tr><td>skew</td><td>any</td><td>Shifts the reported timestamp without delaying delivery, so the receiver misreads the age</td><td>|skew|  $\leq 3 0 \mathrm { s }$ </td></tr><tr><td colspan="4">Semantic-content faults</td></tr><tr><td>freeze</td><td></td><td>SPaT, BSM Holds phase, countdown, or kinematics at its value when the window window only</td><td></td></tr><tr><td>rebind</td><td>MAP</td><td>opens Lanes of one movement report the signal group of another</td><td>movement pair</td></tr><tr><td>pos_perturb BSM</td><td></td><td>Position noise σ plus a constant along-track bias b</td><td> $\sigma \in \left[ 0 . 2 , 1 0 \right] \mathrm { m } ,$   $b \in [ \dot { 0 } , 4 0 ] \dot { \mathrm { m } }$ </td></tr></table>

The monitor that identifies divergence episodes uses the budgets of Table 1. A signal belief is in error when its phase differs from the true phase or its countdown differs by more than 1.0 s; a neighbor track is displaced when its position error exceeds the position budget plus its age multiplied by the vehicle’s speed; and a believed track whose vehicle no longer exists becomes a ghost after 2.0 s.

Table 6: Decision rules and violation criteria of the six safety applications.
<table><tr><td>App.</td><td>Decision rule</td><td>Violation</td></tr><tr><td colspan="3">Signal applications</td></tr><tr><td>RLVW</td><td>Predicts the phase at arrival; once  $d \leq d _ { s } ( v )$  , stops if arrival falls on red, or on yellow after the countdown expires. Holds on a current red within 6 m of the bar; without signal data, stops at</td><td>Red entry</td></tr><tr><td>GLOSA</td><td>the decision point Keeps speed if arriving within the current green; otherwise advises  $v = d / t _ { \mathrm { g r e e n } } .$  , clipped to [3, 16] m/s</td><td>Red entry</td></tr><tr><td colspan="3">Conflict applications</td></tr><tr><td>IMA</td><td>Stops if a crossing track reaches the conflict point within 3.5 s of the ego (ego arrival within 12 s)</td><td>Collision, &lt; 1.8 m</td></tr><tr><td>LTA</td><td>Stops if an opposing track arrives before the ego&#x27;s arrival plus a 4.0 s gap (12 s horizon)</td><td>Collision, &lt; 1.8 m</td></tr><tr><td>FCW</td><td>Brakes if the time to collision with the lead is below 2.5 s; holds behind a stopped lead within 10m</td><td>Forward collision, &lt; 2.0 m</td></tr><tr><td>EEBL</td><td>Brakes on a relayed hard-braking flag from a lead vehicle within 80 m</td><td>Forward collision, &lt; 2.0 m</td></tr></table>

## Appendix E Failure Margin and Search Settings

The failure margin $m ( S )$ $\operatorname { E q . } \left( 9 \right)$ measures how close an executed, non-admitted scenario came to failing; smaller is closer. For signal applications it is the smallest non-negative hazard slack over all divergence episodes,

$$
m ( S ) = \operatorname* { m i n } _ { e \colon \delta _ { e } \geq 0 } \delta _ { e } , \qquad \delta _ { e } = \operatorname* { m i n } _ { t _ { d } \in \mathcal { U } _ { e } } \big ( t _ { d } - t _ { e } ^ { \mathrm { e n d } } \big ) ,\tag{34}
$$

where $\mathcal { U } _ { e }$ is the set of decision epochs $t _ { d }$ of the message-driven rollout that occur after episode e begins, and $t _ { e } ^ { \mathrm { e n d } }$ is the episode’s end. A small slack means the wrong belief ended just before a decision; a negative slack means a decision was taken while the belief was wrong. For conflict applications, $m ( S )$ is the minimum distance between the message-driven ego and any agent over the run, counting only vehicles ahead of the ego for FCW and EEBL. If no episode has non-negative slack (including a run without episodes) or no conflict occurs, $m ( S )$ is set to a large constant; a run whose episodes all have negative slack therefore keeps this constant.

Specification-space search starts from the compiled scenario and keeps an elite set of the $K = 8$ highest-priority executed scenarios. Each iteration samples a parent with probability proportional to w(S) (Eq. (26)); a scenario in which both rollouts violate is non-attributable and receives the fixed priority 0.05 instead, the weight of a scenario with $m ( S ) = 1 9 . 5$ , so it remains a possible parent but ranks below every closer near miss. One of the operators in Table 7 is drawn, with weights that favor interaction-timing operators for conflict applications and fault operators for signal applications. Mutations never change the application, ego maneuver, agent roster, or map query, so the test intent is preserved. Each mutant passes through the verifier before execution, and the budget of $N _ { s } = 1 2 0$ candidates per scenario counts both executed and gated mutants. Admitted scenarios remain in the elite set so that search can explore neighboring mechanism cells. Each admitted scenario is minimized by delta debugging (Section 2.5). The refinement study runs search with a fixed random seed on the stub backend, and allows up to $\bar { N _ { r } } = \bar { 5 }$ reflection rounds per scenario.

Table 7: Mutation operators of specification-space search. <sup>∗</sup>Replaces the scenario’s fault vector.
<table><tr><td>Operator</td><td>Change</td></tr><tr><td>Fault vector</td><td></td></tr><tr><td>Fault magnitude</td><td>Scales a delay or skew by ×0.5 or ×2; changes a drop count by —5 to +10; scales σ or shifts b</td></tr><tr><td>Fault window</td><td>Shifts the offset by ±2.5 s, resets the duration, or changes the anchor event</td></tr><tr><td>Stuck feed*</td><td>SPaT phase and countdown freeze opening 2.5 or 5 s before the amber onset</td></tr><tr><td>Outage* Track freeze*</td><td>Window-long drop of the consumed stream (5, 8, or 12 s)</td></tr><tr><td>Position bias*</td><td>Freezes neighbors’ BSM kinematics from 3–8 s before the ego&#x27;s decision point</td></tr><tr><td></td><td>Persistent BSM bias of 8, 15, or 25 m  $( \sigma = 0 . 5 \mathrm { m } )$ </td></tr><tr><td>Pathology swap*</td><td>Library pathology at a random intensity</td></tr><tr><td>Interaction and equipage</td><td></td></tr><tr><td>Conflict alignment Agent kinematics</td><td>Places a conflict agent so that it arrives with the ego, to within ±8%</td></tr><tr><td>Ego speed</td><td>Shifts an agent&#x27;s distance by 5–40 m within [5, 300] m, or its speed by 0.5–3 m/s within [0.5, 35] m/s</td></tr><tr><td>Penetration</td><td>Shifts the approach speed by ±2 m/s within [2, 25] m/s</td></tr><tr><td>SPaT rate</td><td>Shifts the market penetration rate by ±0.25 within [0, 1]</td></tr><tr><td></td><td>Halves or doubles the SPaT rate within [1, 20] Hz</td></tr></table>

## Appendix F LLM Prompts

LAST-V2X uses three LLM roles that share one backend, GPT-4o at temperature 0.2 with a fixed sampling seed (Section 3). Each receives a fixed system prompt, condensed in Fig. 8, and must return a single JSON object that is validated before use. Rules are paraphrased, and the worked examples in the parser prompt are omitted. The domain notes given to the reflection role come from the V2X knowledge base and are identical for every scenario; they are not tuned to individual prompts.

## Appendix G Reproducibility

Table 8 summarizes the configuration of each experiment. All runs are deterministic given a scenario seed, which fixes vehicle equipage, message-bus jitter, fault stochasticity, RSU pole selection, vehicle blueprints, and retrieval tiebreaking, so every scene replays identically. Each admitted failure is stored with its full minimal-witness specification and re-verified against the paired admission test on replay. For every benchmark scene, LAST-V2X archives the compiled specification, the scene plan with its rationale, the grounding record, the execution trace, the monitor report, three camera streams with rendered V2I/V2V links, and RSSI, latency, and delivery maps. Further material is available on the project page: https://keshuw95.github.io/LAST-V2X/.

Table 8: Experiment configurations. All runs use GPT-4o (temperature 0.2, fixed sampling seed) and synchronous live CARLA simulation, except <sup>‡</sup>search runs, which use the stub backend; the headline mechanisms were validated in live CARLA.
<table><tr><td>Experiment</td><td>Scale</td><td>Seeds</td></tr><tr><td>Normal-traffic benchmark</td><td>100 prompts, 8 towns</td><td>1 per scene</td></tr><tr><td>Communication-fault suite</td><td>12 prompts</td><td>1 per scene</td></tr><tr><td>Corridor traversal</td><td>6 prompts, Town05; 3 offset plans × 3 radio ranges</td><td>10 (420 runs)</td></tr><tr><td>Multiple connected vehicles</td><td>6 prompts, Town05</td><td>10 per condition</td></tr><tr><td>Refinement suite</td><td>50 prompts; ≤120 search candidates and ≤5 reflection rounds per scene</td><td>1 per scene</td></tr></table>

<table><tr><td>Parser</td><td></td></tr><tr><td>ROLE</td><td>Translate a natural-language V2X test intent into a scenario-specification JSON object for a CARLA stress-testing pipeline that evaluates SAE Day-1 safety applications under communication degradation. The six applications are listed with their descriptions and preconditions; choose exactly one.</td></tr><tr><td>RULES</td><td>(i) If the prompt describes only traffic, infer the application from the conflict (left turn with oncoming traffic → LTA; crossing vehicle → IMA; slower or stopped vehicle ahead → FCW; relayed hard braking → EEBL; signalized approach about arriving on green → GLOSA; other signalized approaches → RLVW) and assume a healthy, standards-nominal channel (SPaT 10 Hz, MAP 1 Hz, BSM 10 Hz, full penetration, no faults); never invent a degradation. (ii) Transcribe every vehicle mentioned, including background traffic.</td></tr><tr><td></td><td>(iii) Time each conflict so that the agent&#x27;s distance divided by its speed matches the ego&#x27;s arrival. (iv) Express degradations only with the seven fault primitives and anchored windows; a pathology library describes known patterns in these terms. (v) A fault must be able to breach the application&#x27;s age budget: a drop of k messages at rate f breaches budget only if  $k / f > B _ { a }$  . A plain SPaT outage is usually fail-safe, so a stuck feed is expressed as a freeze of phase and countdown opened before the phase change.</td></tr><tr><td>REPAIR</td><td>(vi) Keep the 6/3/12 s signal plan, the driver-character presets, the weather, and the channel defaults unless the prompt states otherwise. (vii) Respond with only the JSON object.</td></tr><tr><td>Planner</td><td>A schema violation or blocking verifier finding is returned as a follow-up message (“That JSON failed validation: ...&quot; or “The spec was rejected by the feasibility verifier: .. .&quot;) asking for a corrected object, for up to three retries.</td></tr><tr><td>ROLE</td><td>Place the specified agents at a junction that is already bound, given live junction facts: the ego&#x27;s approach, the fixed signal cycle, and the candidate approaches for each conflict role with their true lane headings.</td></tr><tr><td>RULES</td><td>(i) Keep the same agent identifiers and do not add or remove agents. (ii) Assign crossing and oncoming agents to one of their role&#x27;s candidate approaches; agents ahead, behind, or in the background share the ego&#x27;s lane.</td></tr><tr><td>OUTPUT</td><td>(iii) Time each conflict so that the agent arrives within a few seconds of the ego. (iv) Respect each agent&#x27;s character: stopped vehicles stay stopped, aggressive agents may run the red, and other agents comply with their signal.</td></tr><tr><td>Reflection</td><td>A JSON plan with a rationale, the ego approach speed, and per-agent approach, distance, speed, and behavior; invalid plans are returned with the problems listed.</td></tr><tr><td>ROLE</td><td>Module IV Act as the refinement analyst. A prompt was compiled and executed as a paired rollout; a scenario is admitted only if the message-driven ego violates its safety criterion while the ground-truth ego does not. Given the prompt, the</td></tr><tr><td>EVIDENCE DIGEST RULES</td><td>evidence digest, and a goal, diagnose why the goal was not met and revise the prompt. Admission status; each rollout&#x27;s violation and minimum conflict gap; the active faults; divergence episodes with onset, end, hazard slack, and whether they are structural; AoI at the decision epochs; and verifier findings.</td></tr><tr><td></td><td>(i) Keep the scenario&#x27;s identity (place, conflict story, application) and change only timing, distances, speeds, fault strength or anchoring, equipage, or wording the parser misread.</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>(ii) Write plain English, not JSON or parameter names.</td></tr><tr><td></td><td></td></tr><tr><td></td><td>(iii) Change as little as plausibly moves the run toward the goal, and escalate decisively if an earlier revision</td></tr><tr><td></td><td>changed nothing.</td></tr><tr><td>DOMAIN</td><td>• Delivery faults make the belief stale, which the age budget detects, so applications usually fall back safely;</td></tr><tr><td>NOTES</td><td>content faults keep a wrong belief looking fresh.</td></tr><tr><td></td><td></td></tr><tr><td></td><td>• A frozen signal matters only if it masks a phase change and persists through the decision.</td></tr><tr><td></td><td></td></tr><tr><td></td><td>• A conflict matters only if the arrivals coincide and the vehicle is within V2V range early enough.</td></tr><tr><td></td><td></td></tr><tr><td></td><td>• A persistent position bias is more potent than noise of the same size.</td></tr><tr><td></td><td></td></tr><tr><td></td><td>• Describing a feed as “stuck&quot; compiles to a persistent hold, whereas “freezes for a few seconds&quot; compiles to a</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>OUTPUT</td><td></td></tr><tr><td></td><td>short one.</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>A JSON object with a diagnosis citing the evidence, the list of changes, and the full revised prompt, which re-enters</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>Module I.</td><td></td></tr></table>

Figure 8: Condensed system prompts of the three LLM roles of LAST-V2X.