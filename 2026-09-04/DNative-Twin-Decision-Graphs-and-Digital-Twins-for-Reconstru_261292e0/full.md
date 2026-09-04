# DNative-Twin: Decision Graphs and Digital Twins for Reconstructable Agentic Decisions

Junjie Pang<sup>1</sup>, Zhenzhen Xie<sup>2</sup>, Haoke Han<sup>1</sup>, Ying He<sup>3</sup>, Jing Wang<sup>4</sup>, Gang Liu<sup>4,†</sup>

<sup>1</sup>Qingdao University, Qingdao, China <sup>2</sup>Shandong University, Qingdao, China <sup>3</sup>Hangzhou Shujiao Technology Partnership (Limited Partnership), Hangzhou, China <sup>4</sup>Changchun University of Technology, Changchun, China

Abstract—AI agents increasingly gather evidence, invoke tools, apply constraints, and produce decisions that people or software may commit to action. A final output alone cannot show which evidence, tool state, rule, authorization, or action path produced it. We present DNative-Twin, a graph-native digital twin that records a committed agentic decision as a typed trajectory and re-executes its decision mechanism under declared conditions. The graph links the state observed by the agent, the path it followed, and the authority behind the resulting action. The twin synchronizes this information, replays the mechanism in isolation, and compares it under controlled changes. We instantiate the framework in enterprise decision processes using three public process logs and controlled replay suites. The experiments identify a specific failure: graph structure localizes represented changes but cannot determine the consequence of an unobserved tool state. In a three-condition controlled experiment with 300 injected instances, unresolved-divergence recall increased from 0 to 0.667 when replay-contract state was added and to 1.0 when verification results were also available; the held-out set contained no critical-class instance. Across 500–5,000 BPI 2020 cases, median end-to-end time increased from 0.794 to 8.889 seconds on the reported platform. These results separate the roles of graph structure, replay context, and verification evidence in reviewing a decision mechanism.

Index Terms—AI agents, agentic decisions, decision reconstructability, digital twin, heterogeneous graph, agent governance

## I. INTRODUCTION

AI agents now gather evidence, invoke tools, apply constraints, and recommend or execute actions in personal, organizational, and public settings. Some decisions remain under direct human review; others operate under authority delegated in advance. In both settings, a consequential action must be connected to what the system observed, how it acted, and the authority under which the action was taken.

Consider two procurement runs that both return APPROVE. In the first, the agent reads a current supplier alert, applies the valid purchasing policy, receives a successful budget response, and sends the recommendation to an authorized manager. In the second, the alert is stale, the budget call times out, and an informal override reaches the same label. The outputs agree, but the decision mechanisms differ. These differences determine whether the committed action can be explained, reproduced, and safely revised. We call the ability to recover this path decision reconstructability.

Existing operational and provenance records rarely preserve this path as one connected record. They often leave three parts disconnected: what the agent observed, how it acted, and what authorized the resulting action. Decision reconstruction requires these dependencies to be linked over time and connected to the observed outcome.

We represent this information as a decision trajectory graph: a dynamic heterogeneous subgraph bounded to one decision path. A real trajectory may end in commitment, while a replay may instead end in review or rejection. The graph supports reconstructability checks, constrained-path queries, and aligned comparison across real, twin, and perturbed environments. A GraphDiff records the differences between two aligned trajectories. The synchronized state and replay conditions provide the context needed to interpret those differences.

We implement this model in DNative-Twin, a graph-native digital twin for reconstructing and testing agentic decision mechanisms(Fig. 1). The framework first copies the relevant state into an isolated twin and replays the recorded mechanism under fixed conditions. It then changes selected conditions and compares the new trajectory with the baseline. The resulting GraphDiff shows where the mechanism changed. A rule-based review then determines whether that difference changes the decision under its declared constraints. Any proposed revision remains linked to the evidence that motivated it and must satisfy declared release conditions.

This workflow requires an identifiable decision boundary, recordable dependencies, explicit authorization, and an observable or pending outcome. Enterprise processes provide these conditions through persistent event records, explicit decision objects, and approval paths, so we first evaluate the framework in this setting. Public logs test whether the graph can be constructed from recorded events. Controlled experiments then test replay, perturbation, adjudication, and revision. In one injected suite, a graph-only detector classified all 40 unobserved tool-timeout cases as benign. In the threecondition experiment, replay-contract state recovered 60 of 90 unresolved instances, and verification results recovered all 90.

This paper makes four contributions:

1) We define a dynamic heterogeneous decision graph that connects what an agent observed, how it acted, and how the resulting action was authorized and committed.

2) We derive decision trajectories as decision-bounded subgraphs and formalize reconstructability for human–AI and delegated automated decisions.

3) We present DNative-Twin, which replays a decision mechanism in synchronized state and compares its trajectory under controlled changes.

4) We provide an executable specification and evaluate its enterprise instantiation on public process logs and controlled replay suites.

Section II defines the decision graph, trajectory, and reconstructability conditions. Sections III and IV present the digitaltwin architecture and its executable specification. Section V combines the enterprise case study with controlled evaluation. Sections VI–VIII discuss related work, scope, limitations, and conclusions.

## II. AGENTIC DECISIONS, DECISION GRAPHS, AND RECONSTRUCTABILITY

This section defines the information needed to reconstruct an agentic decision. It first represents the decision context as a dynamic heterogeneous graph. It then defines the trajectory of a committed decision, the conditions for reconstructability, and the comparison of trajectories across environments.

## A. Decision Graph

We first define the state required to distinguish the two procurement runs introduced above. Let the decision-relevant state of an agentic system at time t be

$$
S _ { t } = ( B _ { t } , K _ { t } , P _ { t } , T _ { t } , H _ { t } , A _ { t } , O _ { t } ) ,\tag{1}
$$

where $B _ { t }$ contains decision objects; $K _ { t }$ contains versioned knowledge; $P _ { t }$ contains policies, constraints, and authority grants; $T _ { t }$ contains tools and their observable states; $H _ { t }$ contains authorized human or machine actors and their roles; $A _ { t }$ contains agents and bounded skills; and $O _ { t }$ contains outcome signals. $S _ { t }$ is the state relevant to one class of decisions. It includes an entity or state when changing that element can alter what an agent or authorized actor should know, do, avoid, escalate, commit, or later revise.

The decision layer is modeled as a temporal attributed heterogeneous multigraph

$$
\mathcal { G } _ { t } = ( V _ { t } , E _ { t } , X _ { t } , \phi , \psi , \rho ) ,\tag{2}
$$

where $V _ { t }$ is the set of nodes observed or valid by time $t ,$ $E _ { t } \subseteq V _ { t } \times { \mathcal { T } } _ { E } \times V _ { t }$ is the set of typed directed edges, and $X _ { t }$ contains node and edge attributes. The maps $\phi : V _ { t } \to \pi _ { V }$ and $\psi ~ : ~ E _ { t } ~  ~ \mathcal { T } _ { E }$ assign node and relation types. The metadata map $\rho$ assigns each graph element an environment label, source time, ingestion time, validity interval, version, provenance reference, and confidence when applicable. A multigraph is required because the same entities may participate in several relations or repeated time-indexed interactions.

The graph contains state, execution, authorization, and decision entities. An interaction is represented by nodes and typed edges instead of one overloaded event tuple. For example, a ToolCall node connects to its invoking Agent, input DecisionObject, used EvidenceArtifact, tool version, and produced result. A CommitmentEvent connects a Recommendation to the AuthorizedActor, DecisionRole, and AuthorityGrant that turn it into an action with consequences. Later OutcomeSignal nodes record the observed result. This factorization preserves shared dependencies: one policy can govern several actions, one evidence artifact can support several recommendations, and one authority grant can cover a bounded class of commitments.

The state tuple identifies what must be retained; the graph schema makes those dependencies queryable. Table I gives the smallest set of types used by the invariant checks. A domain may add or refine types while preserving the distinction between context, execution, authorization, and commitment.

Each node used in a decision trajectory records its identity, time, type, environment, and source. Additional metadata is attached only when it is needed to reconstruct a dependency: version references for evidence, execution references for tool calls, and authority references for commitments. The graph may store governed references instead of sensitive source content.

This schema connects what informed a decision, how the mechanism acted, and what authorized the commitment. The procurement case later maps these roles to purchase requests, managers, and approvals. In an automated setting, the same authority path can end at a machine actor operating within a grant issued in advance.

## B. Decision Trajectory and Reconstructability

A decision mechanism is denoted by

$$
{ \mathcal { M } } = ( Q , \pi , \Gamma , \Lambda ) ,\tag{3}
$$

where $Q$ is a set of typed decision states, π is the action policy of participating agents and actors, Γ contains transition and path constraints, and Λ defines commitment, escalation, and release rules. Keeping M separate from $\mathcal { G } _ { t }$ allows the same mechanism to be replayed against different synchronized states and allows candidate mechanisms to be compared over the same decision object. We decompose the constraint set as

$$
\Gamma = \Gamma _ { \mathrm { p a t h } } \cup \Gamma _ { \mathrm { t e m p o r a l } } \cup \Gamma _ { \mathrm { v e r s i o n } } \cup \Gamma _ { \mathrm { a p p } } ,\tag{4}
$$

where $\Gamma _ { \mathrm { a p p } }$ contains state-conditioned applicability constraints.

DNative-Twin distinguishes three properties that are often conflated. Traceability establishes the source, provenance, and version of an artifact. Validity establishes whether that artifact was legally or temporally valid when used. Contextual applicability establishes whether a valid artifact applies to the current decision state. For a state-sensitive dependency v, we define

$$
\begin{array} { r } { \mathrm { A p p } ( v , \mathcal { S } _ { t } ) = \mathbf { 1 } [ S _ { t } \mid = A _ { v } ] , ~ } \\ { A _ { v } = \mathrm { s c o p e } ( v ) ~ } \\ { ~ \land \mathrm { p r e c o n d i t i o n } ( v ) . } \end{array}\tag{5}
$$

Thus, Valid $\mathbf { \Phi } ( v , t ) = 1$ does not imply $\mathrm { A p p } ( v , S _ { t } ) = 1$ . A purchasing rule may remain the current valid version while applying only below a monetary threshold or outside a highrisk supplier class. The metadata map $\rho$ therefore includes machine-readable scope and precondition fields for statesensitive policies, evidence, authority grants, skills, tools, and path rules [1]. Whenever natural-language conditions cannot be compiled into explicit predicates, the graph records the unresolved condition and routes the dependency to governed review rather than treating it as applicable by default.

TABLE I  
MINIMUM NODE AND RELATION TYPES IN THE DECIS ION GRAPH.
<table><tr><td>Category</td><td>Minimum node types</td><td colspan="2">Representative edge types</td></tr><tr><td>Decision context</td><td>DecisionObject, KnowledgeUnit, PolicyRule</td><td>EvidenceArtifact, about, governs</td><td>supports, contradicts, version_of,</td></tr><tr><td>Execution</td><td>Agent, Skill, ToolCall, DecisionEvent</td><td></td><td>performed, invokes, uses, acts_on, produces</td></tr><tr><td>Authorization</td><td>AuthorizedActor, DecisionRole, ReviewEvent, AuthorityGrant</td><td>delegates_to</td><td>holds_role, authorizes, commits, reviews,</td></tr><tr><td>Decision</td><td>Recommendation,</td><td>CommitmentEvent, recommends,</td><td>committed_as, modified_as,</td></tr><tr><td>Twin operation</td><td>OutcomeSignal ScenarioPerturbation, CandidateRevision</td><td>GraphDiff, perturbs,</td><td>results_in, responds_to replayed_from, diverges_from,</td></tr></table>

For a decision object b, an observation interval $[ t _ { 0 } , t _ { 1 } ]$ and a terminal node z associated with $b ,$ define the decision trajectory as

$$
\begin{array} { r } { \mathcal { T } ( b , z ; [ t _ { 0 } , t _ { 1 } ] ) = \mathcal { G } _ { t } [ V _ { b , z } , E _ { b , z } ] , } \end{array}\tag{6}
$$

where φ(z) ∈ {DecisionEvent, CommitmentEvent, R The set $V _ { b , z }$ contains $b , z ,$ all admissible decision ancestors of z in the interval, and any outcome nodes explicitly attributed to z. The set $\boldsymbol { E _ { b , z } }$ contains the typed relations among those nodes. Admissibility is defined by relations such as uses, supports, invokes, produces, reviews, authorizes, commits, committed\_as, and results\_in. The boundary rule excludes records that merely share a case identifier but have no decision relation to z.

For sequential inspection, the trajectory can be viewed as a time-ordered event sequence:

$$
\tau _ { b , z } = \mathrm { T o p o T i m e } ( \mathcal { T } ( b , z ; [ t _ { 0 } , t _ { 1 } ] ) ) ,\tag{7}
$$

where TopoTime returns a time-respecting ordering with stable tie-breaking for concurrent events. This ordered view is useful for inspection, while the graph retains shared evidence, competing recommendations, parallel tool calls, and reviews that resolve several branches.

A recommendation and a consequential commitment are distinct nodes. Let R be the set of Recommendation nodes and C the set of CommitmentEvent nodes. For $a \in C$ , the committed trajectory $\mathcal { T } ( b , a ; [ t _ { 0 } , t _ { 1 } ] )$ exists only when the graph contains a recommendation relation and an applicable authority path. All antecedent relations point toward the commitment event:

$$
\begin{array} { r l } & { \mathrm { C o m m i t t e d } ( T , a ) = 1 } \\ & { \Longleftrightarrow } \\ & { \mathrm { ~ } q r \in R : ( r , \ell , a ) \in E _ { b , a } , } \\ & { \ell \in \{ \mathrm { c o m m i c t ~ } \mathrm { t e } { \mathrm { e } } { \mathrm { d } } _ { - } { \mathrm { a } } \mathrm { s } , \mathrm { m o d i f ~ } \mathrm { i } \mathrm { e } { \mathrm { d } } _ { - } { \mathrm { a } } \mathrm { s } \} , } \\ & { \exists h , g : ( g , \mathrm { a u t h o r i } \mathrm { z } { \mathrm { e } } { \mathrm { s } } , h ) \in E _ { b , a } , } \\ & { ( h , \mathrm { c o m m i t s } , a ) \in E _ { b , a } , } \\ & { \phi ( h ) = \mathrm { a u t h o r i } \mathrm { z e d } \lambda \mathrm { c t o r } , \quad \phi ( g ) = \mathrm { a u t h o r i t y G r a n t } , } \\ & { \mathrm { V a l i d } ( g , a ; i m e ) = 1 , } \\ & { \mathrm { A p p } ( g , S _ { a } , t _ { i m e } ) = 1 . } \end{array}\tag{8}
$$

This definition records an agent recommendation and the action that gives it practical effect as separate events. In human–AI collaboration, an authorized person may accept or viewEvent}.<sub>modify</sub> <sub>the</sub> <sub>recommendation</sub> <sub>and</sub> <sub>create</sub> <sub>a</sub> <sub>commitment</sub> <sub>event.</sub> A rejection is recorded as a ReviewEvent and produces no commitment event. In automated operation, an authorized machine actor may commit the output within authority granted in advance. Both forms record the source, scope, validity, and constraints of that authority.

With the graph and trajectory established, the technical problem can now be stated. For a commitment event a, let $\mathcal { R } _ { p r e } ( a ) = \{ R _ { 1 } , . . . , R _ { m } \}$ be the required antecedent type groups, where each $R _ { j } \subseteq \mathcal T _ { V }$ contains acceptable alternatives for one requirement. A minimal configuration contains the groups {DecisionObject}, {EvidenceArtifact}, {KnowledgeUnit, PolicyRule},

{Agent, ToolCall}, {AuthorizedActor}, and {AuthorityGrant}. Let C(a) contain the required temporal, version, applicability, cardinality, confidence, and path constraints.

Let $\operatorname { A n c } _ { \mathcal { L } } ( a )$ denote nodes reverse-reachable from a through admissible relation types ${ \mathcal { L } } ,$ and let Desc $\displaystyle \mathbf { \sigma } _ { : \mathcal { L } } ( a )$ denote nodes forward-reachable through admissible outcome relations. Outcome coverage is

$$
\begin{array} { r l } & { \mathrm { O u t c o m e C o v e r e d } ( \mathcal { T } , a ) = 1 } \\ & { \iff [ \exists o \in \mathrm { D e s c } _ { \mathcal { L } } ( a ) : \phi ( o ) = \mathrm { O u t c o m e S i g n a l } ] } \\ & { \qquad \lor \mathrm { P e n d i n g O u t c o m e } ( \mathcal { T } , a ) = 1 . } \end{array}\tag{9}
$$

The pending state records that the outcome is expected but not yet observable. Decision reconstructability is

$$
\begin{array} { r l } & { \mathrm { R e c o n } ( { \mathcal T } , \boldsymbol { a } ) = 1 } \\ & { \Longleftrightarrow [ \forall R _ { j } \in { \mathcal R } _ { p r e } ( \boldsymbol { a } ) , } \\ & { \qquad \exists v \in \mathrm { A n c } _ { \mathcal L } ( \boldsymbol { a } ) : \phi ( v ) \in R _ { j } ] } \\ & { \qquad \land \mathrm { O u t c o m e C o v e r e d } ( { \mathcal T } , \boldsymbol { a } ) = 1 } \\ & { \qquad \land \left[ \forall c \in { \mathcal C } ( \boldsymbol { a } ) : c ( { \mathcal T } ) = 1 \right] . } \end{array}\tag{10}
$$

Given partial observations from heterogeneous systems, the task is to construct a trajectory graph whose commitment events satisfy Recon, preserve the separation between recommendation and commitment, and remain comparable across real, twin, and perturbed environments. This formulation distinguishes three failures. A capture failure occurs when a required entity or relation is absent from the graph. A mechanism failure occurs when a recorded trajectory violates a decision constraint. A divergence occurs when two otherwise valid trajectories differ across environments. The resulting record supports later analysis of correctness, fairness, and causality, each of which requires its own evidence.

Reconstructability establishes what must be present in one trajectory. Comparing the same mechanism across environments additionally requires synchronized state and aligned trajectories. The synchronization contract defines a typed mapping

$$
\sigma : \Delta \mathcal { G } _ { r e a l } ( t ) \to \Delta \mathcal { G } _ { t w i n } ( t ^ { \prime } ) ,\tag{11}
$$

where a real graph delta is normalized, validated, and projected into the twin with explicit source and ingestion times. The mapping need not preserve identity: production identifiers may be tokenized, sensitive evidence may be represented by governed references, and unavailable tools may use consequence-preserving stubs. The contract must state what is synchronized, with what freshness, under which transformations, and how failed updates are represented.

A scenario operator $W _ { \delta } ( \mathcal { G } _ { t w i n } , \mathcal { M } , b )$ applies a typed perturbation $\delta$ and produces a terminal node $z ^ { \delta }$ and trajectory $\mathcal { T } _ { b , z ^ { \delta } } ^ { \delta }$ . The terminal node may be a CommitmentEvent, ReviewEvent, or another declared DecisionEvent. Terminal nodes from two environments are aligned when they share the decision object, replay lineage, mechanism version, and observation window. We write this condition as Align $( z ^ { i } , z ^ { j } \mid b , \mathcal { M } , [ t _ { 0 } , t _ { 1 } ] ) = 1$ ; the terminal-node identifiers and types may differ. For aligned trajectories, the graphdifference operator produces

$$
D ( \mathcal { T } _ { b , z ^ { i } } ^ { i } , \mathcal { T } _ { b , z ^ { j } } ^ { j } ) = ( \Delta V , \Delta E , \Delta X , \Delta P )\tag{12}
$$

and compares nodes, edges, attributes, and decision paths across environments [2]. Let $d _ { i j } ~ = ~ D ( T _ { b , z ^ { i } } ^ { i } , T _ { b , z ^ { j } } ^ { j } )$ denote the resulting GraphDiff. Synchronization projects observed state changes into the twin, perturbation applies a declared intervention, and $d _ { i j }$ records the resulting trajectory changes. Environment labels keep perturbed data separate from realworld evidence and preserve each divergence for review.

Locating a structural change is not enough to determine its governance effect. The next definitions separate structurally different but governance-equivalent trajectories from changes that alter a required policy, authority, evidence, or outcome condition.

Two executions may retrieve independent evidence in different orders while preserving the same decision dependencies. We define governance equivalence, $\mathcal { T } _ { i } \sim _ { \mathcal { Z } } \mathcal { T } _ { j }$ , when the trajectories have equivalent required-evidence coverage, applicablepolicy set, authority path, critical-invariant vector, commitment class, and outcome constraints. Structural differences inside an equivalence class are benign; differences that change one of these governance-relevant projections are material.

For diagnosis, each change in $d _ { i j }$ may receive one or more observable source labels:

$$
\begin{array} { r } { \mathrm { S r c } ( d _ { i j } ) \subseteq \{ \mathrm { s t a t e , c o n f i g , t o o l , s t o c h , m e c h a n i s m } \} . } \end{array}\tag{13}
$$

The source labels are not mutually exclusive. State labels cover evidence, knowledge, policy, authority, or outcome changes. Configuration labels cover model, prompt, skill, seed, or decoding changes. Tool labels cover version, availability, replay mode, or returned state. Stochastic labels mark residual variation under a fixed replay contract, and mechanism labels mark an intentional revision.

Let I be the declared invariants and P the decision policy. A rule-based adjudication function

$$
\left. \begin{array} { l } { \mathrm { C l a s s i f y } ( { d _ { i j } } , { \mathcal { T } } , { \mathcal { P } } , S _ { t } ) } \\ { \displaystyle ~ \in \left. \begin{array} { l } { { b e n i g n , } } \\ { { c r i t i c a l , u n r e s o l \nu e d \ y } . } \end{array} \right. . } \end{array} \right.\tag{14}
$$

returns the first applicable label in the following order. It returns unresolved when the available evidence cannot support a governed classification. With sufficient evidence, it returns critical for a mandatory authority, isolation, evidence, or review violation; material for another change to a governancerelevant projection; and benign when the trajectories remain governance-equivalent. This priority makes the four labels mutually exclusive. Frequency does not override a mandatory constraint: a less frequent trajectory may be the valid one when it satisfies a required escalation rule.

## III. DNATIVE-TWIN ARCHITECTURE

As summarized in Fig. 1, DNative-Twin organizes the decision-reconstruction workflow into three stages: graph projection and synchronization, isolated replay and perturbation, and semantic adjudication and revision.

## A. Graph Projection and Synchronization

Consider the purchase request used throughout the paper. When a new supplier alert arrives, DNative-Twin records the alert and its effect on the request, then copies that state into an isolated twin. The twin first replays the recorded decision and then tests a declared change, such as a budget-tool timeout. Comparing the two paths shows where the decision mechanism changed and whether the change violates a constraint. If adjustment is needed, the proposed revision is tested under the same conditions before release. Figure 1 shows this closed loop from observation to review.

![](images/c3eaba0e2c2c3398bb22e8c9cdb68541295bfe5e0920dc4dcfc9c0702b7cd959.jpg)  
Fig. 1. Overview of DNative-Twin. (A) Decision-relevant records are projected into a synchronized trajectory graph. (B) The recorded mechanism is replayed in isolation under baseline and declared perturbation conditions, producing aligned trajectory differences. (C) GraphDiffs are adjudicated against declared constraints and supporting evidence; candidate revisions are returned for verification before release, while outcome feedback updates the synchronized state.

Source connectors capture decision-relevant changes from operational systems and agent activity. A normalization step converts each source record into a typed graph update while preserving its origin and time. The graph can therefore connect a decision to the information and authority on which it depended, as well as to its later outcome.

No single source record normally contains this full path. Table II therefore compares the information retained by common record types. DNative-Twin joins these records around a decision trajectory, which becomes the unit of replay and comparison.

## B. Isolated Replay and Perturbation

Isolated replay asks whether the same mechanism follows the same governed path when it runs against a synchronized copy of the decision state. For the purchase request, the twin can reuse the recorded supplier alert, policy version, and budget state without issuing another order. Any call that could affect the real environment is replaced by a read-only or simulated call labeled environment=twin. The twin has no authority to commit an external action.

This isolation is useful only when the real and twin runs refer to the same decision. The replay record therefore binds the real object, mechanism version, synchronization cutoff, and any controlled or substituted execution state. Comparison focuses on the resulting decision path and its governed dependencies. Thus, two runs that both approve a request still differ materially if the replay omits a required supplier alert or review. Different internal traces remain governance-equivalent when they satisfy the same declared constraints.

Under these bound conditions, shadow replay re-executes the mechanism against the synchronized state:

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { s h a d o w } } = \mathrm { E x e c u t e } ( \mathcal { M } , \mathcal { S } _ { t } ^ { \mathrm { t w i n } } , } \\ { \mathcal { C } _ { R } ) . \qquad } \end{array}\tag{15}
$$

The real trajectory $\mathcal { T } _ { \mathrm { r e a l } }$ is an attributable comparison reference rather than executable ground truth. The replay contract

$$
\begin{array} { c } { { { \mathcal { C } } _ { R } = ( b , t _ { \mathrm { c u t } } , K , P , A , } } \\ { { { \mathcal { T } } , H , \Theta , \Omega ) . } } \end{array}\tag{16}
$$

groups the replay conditions into four parts: decision identity and cutoff $( b , t _ { \mathrm { c u t } } )$ ; governed state (K, P, H); execution configuration $( A , T , \Theta ) ;$ and environment Ω. Each unbound or substituted field is recorded as a possible source of divergence.

Tool behavior is the main case in which a replay needs a substitute. The framework supports three modes: reuse the recorded response, execute a read-only copy, or use a stub that preserves the declared result or failure. These modes answer different questions. Reusing a response isolates the downstream mechanism, whereas execution also tests the synchronized tool state. Table III states the interpretation of each mode; the ToolCall node records the selected mode and the information needed to reproduce it.

Replay establishes a baseline; perturbation then tests one declared dependency of that baseline. For example, the purchaserequest replay can replace a successful budget check with a timeout. The resulting ScenarioPerturbation identifies the changed dependency and links it to the trajectory that diverges from the baseline.

Each scenario records the intervention in a machinereadable form and remains tied to the original decision and replay state. This link makes the comparison attributable: the observed difference is evaluated against a named change rather than an unspecified change in the environment.

## C. Semantic Adjudication and Revision

Semantic adjudication determines whether a replay difference affects a declared constraint and whether the mechanism requires adjustment. In the budget-timeout example, a missing escalation path would be classified against the declared invariants. A candidate revision could then add the required fallback for that condition. The graph links the revision to the observed divergence and its verification record [3]. The revision becomes eligible for deployment only after it satisfies the release conditions below.

TABLE II  
CONCEPTUAL COVERAGE OF COMMON RECORDS. “STRONG” INDICATES THAT THE INFORMATION IS A FIRST-CLASS PART OF THE RECORD, NOT THAT EVERY PRODUCT IN THE CATEGORY IMPLEMENTS IT.
<table><tr><td>Capability</td><td>Workflow log</td><td>Agent trace</td><td>Provenance record</td><td>DNative-Twin graph</td><td>trajectory</td></tr><tr><td>Process state and transi- tion</td><td>Strong</td><td>Partial</td><td>Partial</td><td>Strong</td><td></td></tr><tr><td>Model/tool execution path</td><td>Weak</td><td>Strong</td><td>Partial</td><td>Strong</td><td></td></tr><tr><td>Evidence and knowledge version</td><td>Partial</td><td>Partial</td><td>Strong</td><td>Strong</td><td></td></tr><tr><td>Authorization and com- mitment</td><td>Partial</td><td>Weak</td><td>Partial</td><td>Strong</td><td></td></tr><tr><td>Outcome feedback Synchronized shadow re-</td><td>Partial</td><td>Weak</td><td>Partial</td><td>Strong</td><td></td></tr><tr><td>play</td><td>No</td><td>Partial</td><td>No</td><td>Yes</td><td></td></tr><tr><td>Typed perturbation and cross-environment diff</td><td>No</td><td>No</td><td>No</td><td>Yes</td><td></td></tr></table>

TABLE III  
TOOL-CALL REPLAY MODES AND INTERPRETATION
<table><tr><td>Mode</td><td>Question answered</td></tr><tr><td>Response injection</td><td>Does the downstream mechanism reproduce its gov- erned behavior under the historical observation?</td></tr><tr><td>Executable replay</td><td>How does the mechanism behave under the currently synchronized tool state?</td></tr><tr><td>Consequence- preserving stub</td><td>How does the mechanism handle a declared result or failure without production side effects?</td></tr></table>

Release readiness tests whether a candidate mechanism $\mathcal { M } ^ { \prime }$ remains within its declared constraints. The mandatory scenario suite is $\mathcal { D } _ { c r i t } .$ . Its pass indicators are $C _ { i n v }$ for invariants and $C _ { s c n }$ for critical scenarios. The remaining terms measure material divergence $\left( r _ { m a t } \right)$ , governance equivalence $( r _ { e q } )$ , twin freshness $( f _ { t w i n } )$ , and unresolved critical cases $( N _ { c r i t , u n r e s } )$ against configured thresholds. We define

$$
\iff \left\{ \begin{array} { l l } { C _ { i n v } = 1 , \quad C _ { s c n } = 1 , } \\ { r _ { m a t } \leq \epsilon _ { m a t } , \quad r _ { e q } \geq \tau _ { e q } , } \\ { f _ { t w i n } \geq \tau _ { f r e s h } , \quad N _ { c r i t , u n r e s } = 0 . } \end{array} \right.\tag{17}
$$

Passing these criteria makes the mechanism eligible for authorization under the declared scenario suite and thresholds. The subsequent GOVAPPROVES step records the independent authorization decision; release requires both results. Synchronization drift, a policy or tool change, a new incident, or a new critical scenario reopens the revision cycle when it invalidates a release condition. A maximum iteration count and replay budget stop the cycle when the available evidence cannot distinguish candidate mechanisms.

Algorithm 1 summarizes the graph, replay, difference, and adjudication operations defined above. Here, S is the sourcerecord collection and $\Delta _ { s c n }$ is the set of declared perturbations. The observation window $[ t _ { 0 } , t _ { 1 } ]$ is part of every trajectory comparison.

Algorithm 1 Synchronize–Replay–Stress–Diff–Revise Loop   
Require: $S , \sigma , \mathcal { M } , b , [ t _ { 0 } , t _ { 1 } ] , \Delta _ { s c n } , \mathcal { T } , \mathcal { P }$   
Ensure: Review package R   
1: $G _ { r } \gets \mathbf { O B S E R V E } ( S , b )$   
2: $G _ { t } \gets \mathrm { A P P L Y I S O } ( \sigma ( G _ { r } ) )$   
3: $( z _ { r } , T _ { r } ) \gets \mathrm { T R A J E C T O R Y } \big ( G _ { r } , b , [ t _ { 0 } , t _ { 1 } ] \big )$   
4: $( z _ { s } , T _ { s } ) \gets \mathrm { S H A D O W } \left( G _ { t } , \mathcal { M } , b , [ t _ { 0 } , t _ { 1 } ] \right)$   
5: $C _ { s } \gets \mathrm { C H E C K } ( T _ { s } , \mathcal { T } )$   
6: assert $\mathrm { A L I G N } \big ( z _ { r } , z _ { s } \mid b , \mathcal { M } , [ t _ { 0 } , t _ { 1 } ] \big ) = 1$   
7: $D _ { s } \gets \mathrm { D I F F } ( T _ { r } , T _ { s } )$   
8: $L _ { s } \gets \mathrm { C L A S S I F Y } ( D _ { s } , \mathcal { T } , \mathcal { P } , S _ { t } )$   
9: $E  \emptyset$   
10: for all $\delta \in \Delta _ { s c n }$ do   
11: $G _ { \delta } \gets \operatorname { P E R T U R B } ( G _ { t } , \delta )$   
12: $( z _ { \delta } , T _ { \delta } ) \gets \mathrm { S H A D O W } \left( G _ { \delta } , \mathcal { M } , b , [ t _ { 0 } , t _ { 1 } ] \right)$   
13: $C _ { \delta } \gets \mathbf { C } _ { \mathrm { H E C K } } ( T _ { \delta } , \mathcal { T } )$   
14: assert $\mathrm { A L I G N } \big ( z _ { s } , z _ { \delta } \mid b , \mathcal { M } , [ t _ { 0 } , t _ { 1 } ] \big ) = 1$   
15: $D _ { \delta } \gets \mathrm { D I F F } ( T _ { s } , T _ { \delta } )$   
16: $L _ { \delta } \gets \mathrm { C L A S S I F Y } ( D _ { \delta } , \mathcal { T } , \mathcal { P } , S _ { \delta } )$   
17: $E \gets E \cup \{ ( \delta , C _ { \delta } , D _ { \delta } , L _ { \delta } ) \}$   
18: end for   
19: $R \gets \mathrm { R E V I E W } ( C _ { s } , D _ { s } , L _ { s } , E )$   
20: $\mathcal { M } ^ { \prime } \gets \mathrm { R E V I S E } ( R )$   
21: $q _ { r } \gets \mathrm { R E L E A S E R E A D Y } ( \mathcal { M } ^ { \prime } )$   
22: $q _ { g } \gets \mathbf { G o v A P P R O V E S } ( \mathcal { M } ^ { \prime } )$   
23: if $q _ { r } \wedge q _ { g }$ then   
24: RELEASE(M<sup>′</sup>)   
25: end if   
26: return R

## IV. CHECKABLE PROPERTIES AND MINIMAL ARTIFACT

## A. From Architecture to Executable Specification

The architecture becomes technically reviewable when its main claims can be checked. DNative-Twin expresses its declared decision constraints as graph properties. Table IV connects each property to a graph check and a concrete failure. Together, these checks test whether an implementation preserves the distinctions required for reconstruction and replay.

TABLE IV  
CORE INVARIANTS AND THEIR GRAPH-LEVEL CHECKS.
<table><tr><td>Invariant</td><td>Required graph check</td><td>Example violation</td></tr><tr><td>I1. Recommendation- commitment separation I2.</td><td>Recommendation and CommitmentEvent are distinct nodes connected by a typed commitment or modification edge</td><td>Agent output is stored directly as an authorized action</td></tr><tr><td>Reconstructability reachability I3. Required-review non-</td><td>Every commitment event satisfies the antecedent-group and outcome-coverage checks in Recon Every path matching a review predicate passes a valid</td><td>A committed action has no retrievable evidence, authority basis, or outcome linkage A high-risk decision reaches commitment without its required</td></tr><tr><td>bypass I4. Twin isolation</td><td>ReviewEvent before commitment A CommitmentEvent in the twin or perturbed en-</td><td>review Shadow tool call commits a purchase order</td></tr><tr><td></td><td>vironment has no valid AuthorityGrant for an external action Each trajectory dependency uses a version valid at event time</td><td></td></tr><tr><td>I5. Knowledge-version traceability I6. Divergence retention</td><td>and records its content reference Each material real-twin or baseline-perturbed difference is</td><td>Decision cites “current policy&quot; without a resolvable version Replay differs but only the final label is retained</td></tr><tr><td></td><td>stored as a GraphDiff linked to the aligned trajectories by typed relations</td><td></td></tr><tr><td>I7. Contextual applicabil- ity</td><td>Every state-sensitive dependency used by a trajectory satisfies its declared scope and preconditions under the execution state</td><td>A valid low-value purchasing rule is used for a high-value, high-risk request</td></tr></table>

## B. Graph Algorithm Interface

The invariants require six graph operations. Reverse reachability checks the dependencies required by Recon, while constrained path queries verify review and authority paths. Temporal and applicability checks establish whether a recorded dependency was valid for the decision state. Subgraph matching detects prohibited shortcuts, such as a recommendation connected directly to an external action. Graph difference then reports changes between aligned trajectories. A property graph with a rule or query engine is sufficient to implement these operations.

## C. Minimal Artifact Specification

A minimal reproducible artifact should test whether the specification can be executed outside the deployed system. Table V lists the files needed for that test.

The real and twin examples must share a decision object, replay lineage, mechanism version, and observation window. Each graph retains its terminal node, z<sub>r</sub> or z<sub>s</sub>, so that alignment is checked before graph difference. At least one declared perturbation should then change the synchronized trajectory. The checker returns pass/fail results with witness paths or missing-type diagnostics for I1–I7, and the difference tool reports node, edge, attribute, and path changes. The scenario manifest, release conditions, and evaluation summary supply the remaining inputs to ReleaseReady. Together, these files allow an independent implementer to construct trajectories, verify invariants, classify their differences, and execute the declared release checks without access to the deployed system.

## V. ENTERPRISE CASE STUDY AND CONTROLLED EVALUATION

Enterprise processes provide persistent events, identifiable decision objects, explicit approval paths, and observable outcomes. These properties make the domain suitable for showing how the general model is instantiated and for testing its graph, replay, perturbation, and revision operations.

## A. Enterprise Study Design and Evidence Layers

The enterprise evaluation separates four sources of evidence. Public logs test graph construction, while controlled overlays add decision fields that the logs do not contain. Constructed suites isolate replay and release behavior, and external-model probes measure repeated-output variation under recorded request protocols. Each field is labeled as source-observed, derived, overlaid, injected, or model-annotated.

The public-log layer uses OCEL 2.0 Procure-to-Pay [4], [5], BPI Challenge 2020 PermitLog [6], and BPI Challenge 2019 [7]. The logs record process events and object relations. LLM prompts, replay modes, authority grants, and machinereadable applicability rules appear only in the declared overlay or injected layers.

Table VI reports the graph sizes produced by the declared mappings. The BPI 2019 result uses a seed-42 reservoir sample obtained after scanning the full log. An independent operator reran the fixed pipeline on the checksum-verified 728,558,522- byte XES and reproduced all 18 declared checks. The rerun recovered 251,734 traces, 1,595,923 events, the ordered 5,000- case sample, and the graph counts in the table; the sampled case-ID files were byte-identical.

The evaluation asks five questions: whether the graph can represent and retrieve decision dependencies (RQ1), whether replay is stable under a fixed contract (RQ2), whether material and null interventions can be distinguished (RQ3), whether replay-contract and verification information improve adjudication (RQ4), and whether the revision gate rejects an injected defect and accepts its bounded repair (RQ5). The comparison conditions reveal progressively more information: B0 contains only the final output, B1 adds the ordered trace, B2 adds structural GraphDiff, and B3 uses the full graph and its queries. Schema coverage reports which declared types are present; the remaining tests evaluate the behavior of the implementation.

## B. Illustrative Procurement Case Study

We use a constructed procurement case to show how the enterprise instantiation operates. Purchase request PR-2048 concerns a time-sensitive component from supplier S-17.

TABLE V  
MINIMAL ARTIFACT PACKAGE
<table><tr><td>Artifact</td><td>Purpose</td></tr><tr><td>decision_graph_schema.json</td><td>Graph schema and required attributes.</td></tr><tr><td>node_types.json</td><td>Node types and type-specific fields.</td></tr><tr><td>edge_types.json</td><td>Relations and endpoint constraints.</td></tr><tr><td>example_real_graph.jsonl</td><td>Example trajectory from the real environment.</td></tr><tr><td>example_twin_graph.jsonl</td><td>Aligned shadow trajectory from the twin.</td></tr><tr><td>scenario_perturbation.json</td><td>Typed stress-test intervention.</td></tr><tr><td>critical_scenario_suite.json</td><td>Mandatory scenarios used by the release check.</td></tr><tr><td>replay_contract.json</td><td>Bound state, configuration, and tool replay modes.</td></tr><tr><td>applicability_rules.json</td><td>Machine-readable scope and precondition predicates.</td></tr><tr><td>release_conditions.json</td><td>Thresholds, freshness requirement, and replay budget.</td></tr><tr><td>check_graph_invariants.py</td><td>Invariant checks and witness diagnostics.</td></tr><tr><td>graph_diff.py</td><td>Node, edge, attribute, and path differences.</td></tr><tr><td>classify_divergence.py</td><td>Benign, material, critical, or unresolved labels.</td></tr><tr><td>release_readiness.py</td><td>Threshold checks and release-gate evidence.</td></tr><tr><td>evaluation_summary.json</td><td>Scenario results, equivalence rates, freshness evidence, and unresolved-case counts</td></tr><tr><td>README.md</td><td>Construction and execution instructions.</td></tr></table>

TABLE VI  
PUBLIC-LOG GRAPH INSTANTIATION. COUNTS DESCRIBE THE EVALUATED SAMPLES.
<table><tr><td>Dataset</td><td>Cases/objects</td><td>Events</td><td>Nodes</td><td>Edges</td><td>Node types</td><td>Edge types</td></tr><tr><td>OCEL 2.0 P2P</td><td>9,543</td><td>14,671</td><td>30,956</td><td>81,132</td><td>6</td><td>5</td></tr><tr><td>BPI 2020 PermitLog</td><td>5,000</td><td>61,408</td><td>225,290</td><td>349,116</td><td>10</td><td>9</td></tr><tr><td>BPI 2019 reservoir sample</td><td>5,000</td><td>30,428</td><td>103,283</td><td>164,272</td><td>11</td><td>9</td></tr></table>

The request instantiates DecisionObject, the procurement manager instantiates AuthorizedActor, and the recorded approval instantiates CommitmentEvent. In the baseline trajectory, procurement agent A-proc retrieves supplier evidence E-supplier-v12, applies policy P-proc-v7, and receives a successful response from budget tool call TC-budget-91. The agent produces recommendation R-approve-1. Manager H-lee reviews it under authority grant AG-44 and creates commitment event AD-approve-1; the purchasing system later records outcome O-po-issued. This trajectory separates the recommendation from the authorized commitment and its outcome.

Table VII contrasts the baseline record with two replay operations. A call-level execution trace can record input, output, and order. The trajectory graph also retains the dependencies needed to explain why a replay followed a different governed path.

In the perturbed scenario, a regional supplier-risk alert is added as knowledge node K-supplier-alert-v13. The synchronization contract copies its effective time and source reference into the twin. Policy P-proc-v7 remains valid, but the high-risk state activates its escalation rule. Replaying the same request produces R-escalate-2, and the GraphDiff links the changed supplier state to the newly activated path.

The scenario engine next removes the primary manager’s authority. This change exposes a structural defect: the mechanism requires review but has no valid alternate path, so the request remains suspended. The resulting review package links the stalled trajectory to the missing delegation. A candidate revision adds a bounded alternate-approver rule and is replayed against the full scenario suite before release. Two additional scenarios test evidence availability and regional shipping constraints.

The case connects the formal definitions to one complete procurement trajectory. Synchronization introduces a declared knowledge change, shadow execution avoids another purchase order, and perturbation tests a failed condition. The GraphDiff then distinguishes a valid escalation from a missing authority path. The quantitative tests below evaluate the corresponding operations.

## C. Controlled Evaluation Results

## Graph construction and dependency queries.

Table VI shows that the implementation instantiated both case-centric and object-centric logs. On OCEL, all 3,371 evaluable commitments passed recommendation–commitment separation and reconstructability reachability, and all 1,598 purchase-order objects contained a Create–Approve–Pay trajectory. Checks that required an absent field were evaluated only after the field was supplied by the declared mapping layer.

Six mappings specified before the held-out evaluation were applied to the same 5,000 BPI 2020 cases. The base mapping produced 225,290 nodes; strict and broad commitment mappings produced 188,093 and 228,761. Removing the overlay policy layer reduced knowledge-policy traceability from 1.0 to 0, while unresolved resource identity reduced the humanapproval-path pass rate from 1.0 to 0. The strict-evidence variant equaled the base mapping on this sample. The query results therefore depend on preserving the entities and relations required by each question.

Seven controlled cases answered five dependency questions. Overall query accuracy was 0.06 for B0, 0.14 for B1, 0.20 for

TABLE VII  
CALL-LEVEL PROCUREMENT REPLAY: EVENT RECORDING VERSUS EXECUTABLE DECISION-DEPENDENCY ANALYSIS
<table><tr><td>Stage</td><td>Historical log</td><td>Baseline replay</td><td>Perturbed replay and interpretation</td></tr><tr><td>State</td><td>PR-2048, policy v7, supplier evi- dence v12</td><td>Same synchronized snapshot and mech- anism</td><td>Supplier alert v13 or another declared state inter- vention</td></tr><tr><td>Tool call</td><td>check_budget (C18,80000) re- turns sufficient</td><td>Response injection tests downstream reproducibility; executable replay tests synchronized tool state</td><td>Consequence-preserving stub returns timeout without a production write</td></tr><tr><td>Recommendation</td><td>APPROVE</td><td>Governance-equivalent APPROVE is ex- pected under the fixed contract</td><td>ESCALATE is expected if the timeout or alert activates a mandatory fallback</td></tr><tr><td>Governance path</td><td>Manager review under AG-44</td><td>Required evidence, applicable policy, review, and authority remain reachable</td><td>GraphDiff exposes added fallback/review nodes or a missing alternate-authority path</td></tr><tr><td>Interpretation</td><td>Records what occurred</td><td>Estimates residual replay stability under controlled conditions</td><td>Tests whether relevant changes alter the governed trajectory and irrelevant changes do not</td></tr></table>

B2, and 1.00 for B3. B3 executed graph queries over taskspecific cases and overlay ground truth; B0–B2 lacked part of the information available to B3. This experiment shows that the graph queries recover the declared dependencies in the seven constructed cases.

A separate annotation probe tested whether event-log text alone supplied stable governance labels. On 20 BPI 2020 cases, two prompt-separated LLM annotators achieved 0.55 raw agreement (κ = 0.5175).

## Replay, perturbation, and adjudication.

Fifty real BPI 2020 case carriers were executed 30 times under a fixed temperature-zero contract. Exact trajectory match and governance-equivalent replay rate were 1.0, and no trajectory variation was observed in this configuration. A separate injected-noise condition produced a flip rate of $0 . 0 9 8 \pm 0 . 0 1 4$ while preserving governance equivalence. This condition tests whether governance equivalence remains stable when outputs are deliberately changed; external-model variation is measured separately below. Separately, 38 of the 50 selected carriers passed I5, which measures knowledge-version traceability.

The controlled perturbation suite applied 42 material and 27 null interventions to seven constructed cases. Ground truth used an applicability threshold of 50,000, while the detector used an alert band of 55,000 specified before the evaluation. It detected 36/42 material changes (PS=0.8571; F1=0.9231), and all 27 null changes remained stable (NPS=1.0). The six misses were near-threshold changes from 50,000 to 50,001, which identifies the detector’s boundary sensitivity.

We then injected seven drift types into 40 held-out real BPI 2020 carriers, yielding 280 instances. Sensitivity over material and critical interventions, benign stability, and source localization were each 1.0. A graph-only detector classified all 40 unresolved tool-timeout instances as benign, giving zero unresolved recall and macro-F1 0.6667. An unrepresented tool state leaves no structural evidence from which to determine its consequence.

A three-condition controlled experiment specified before the held-out evaluation isolated the required information. Thirty held-out BPI 2020 carriers were combined with 11 declared scenarios. The critical-timeout scenario required an adopted request above 50,000; all 30 carriers were at or below this threshold, so those instances were excluded. The remaining 300 instances comprised 180 benign, 30 material, and 90 unresolved cases, with no critical case. Condition A used graph signals, Condition B added replay-contract tool state, and Condition C additionally used verification-result fields.

TABLE VIII  
SEMANTIC ADJUDICATION UNDER THREE INFORMATION CONDITIONS IN THE CONTROLLED INJECTED PROTOCOL.
<table><tr><td>Metric</td><td></td><td></td><td>A: Graph B: +Contract C: +Verification</td></tr><tr><td>Macro-F1</td><td>0.600</td><td>0.908</td><td>1.000</td></tr><tr><td>Unresolved recall</td><td>0.000</td><td>0.667</td><td>1.000</td></tr><tr><td>Unresolved→benign FN</td><td>1.000</td><td>0.333</td><td>0.000</td></tr><tr><td>Critical support</td><td>0</td><td>0</td><td>0</td></tr></table>

As Table VIII shows, replay-contract state recovered 60 of the 90 injected unresolved instances, and verification-result fields recovered all 90. Condition C uses injected signatureverification results and therefore measures the value of making those outcomes available to adjudication. The reported macro-F1 is averaged over the supported benign, material, and unresolved classes.

The three tool modes clarify why the contract matters. With no drift, response injection, executable replay, and consequence-preserving stubbing agreed. Under budget or supplier drift, injection retained the historical observation, executable replay exposed the current state change, and the stub preserved its decision consequence without a production write. Tool replay mode is therefore part of the meaning of a trajectory comparison.

## Revision, runtime, and external-model probes.

The revision demonstration introduced $\mathcal { M } _ { 0 } .$ , in which a budget timeout had no fallback, and $\mathcal { M } _ { 1 }$ , which added bounded escalation. Critical-invariant pass rate increased from 0.7143 to 1.0, material-divergence rate decreased from 0.3333 to 0, governance-equivalent rate increased from 0.7143 to 1.0, and unresolved critical cases decreased from six to zero. Under the four evaluated thresholds, $\mathcal { M } _ { 0 }$ failed and $\mathcal { M } _ { 1 }$ passed. Applying the gate to a release additionally requires synchronization freshness, the complete critical-scenario suite, iteration limits, and replay-budget checks.

Scalability was measured at four BPI 2020 sample sizes with five repetitions and no failed run. Median end-to-end time was 0.794, 1.682, 4.335, and 8.889 seconds at 500, 1,000, 2,500, and 5,000 cases, respectively. At 5,000 cases, median graph construction and invariant checking required 1.797 and 3.846 seconds. These measurements report observed scaling for Python 3.11.15 and NetworkX 3.6.1 [8] on an Apple M5 Pro with 64 GB RAM.

TABLE IX  
EXTERNAL-MODEL NOISE PROBES ON REAL BPI 2020 CASE CARRIERS.
<table><tr><td>Probe</td><td>Valid calls</td><td>Exact repeat</td><td>Baseline agree.</td></tr><tr><td>DeepSeek-20</td><td>100/100</td><td>0.700</td><td>0.820</td></tr><tr><td>DeepSeek-100</td><td>500/500</td><td>0.870</td><td>0.932</td></tr><tr><td>Kimi-100</td><td>496/500</td><td>0.960</td><td>0.982</td></tr></table>

The external-model probes used five repeated calls per case. Baseline agreement is the fraction of repeated outputs that matched the recorded baseline label. The exploratory 20- case DeepSeek probe and the preregistered 100-case probes produced mostly parseable outputs, with exact-repeat rates from 0.70 to 0.96. Table IX reports the full results.

The 100-case probes used the same carrier set and prompt but different models and provider-imposed sampling parameters. The probes therefore characterize repeated-output variation within each recorded protocol. Four Kimi responses were unparseable, and its interrupted run resumed from per-case checkpoints recorded in metadata. The remaining variation motivates repeated replay when output stability is part of the contract.

The evaluation supports three findings. The declared mappings construct typed graphs from the reported enterprise logs. The controlled suites execute replay, perturbation, adjudication, and bounded constraint adjustment under their specified conditions. In the injected protocols, graph structure misses an unrepresented tool failure, while replay-contract state and injected verification results recover the 90 unresolved instances in the three-condition experiment.

## VI. RELATED WORK

Agent systems interleave model inference with external actions, making tool behavior and environmental state part of the safety boundary. ReAct established a reasoning–acting pattern in which language models alternate between reasoning and actions against external sources [9]. ToolEmu introduced an LM-emulated sandbox for testing high-stakes tools and long-tailed failure scenarios without invoking production services [10]. Agent-SafetyBench broadens this empirical line with interactive environments and tool-mediated safety cases [11]. At the organizational level, the NIST Generative AI Profile recommends continuous monitoring, provenance-aware documentation, and human intervention processes across the AI lifecycle [12].

Sandbox and guardrail work tests whether an action can proceed safely. Decision reconstruction adds a record-level question: which observed state and authorization condition produced the action, and how does that path change under replay? DNative-Twin addresses this question by retaining the governed trajectory and its cross-environment difference.

Provenance systems describe how entities, activities, and agents contribute to the production or transformation of an artifact. The W3C PROV data model provides interoperable concepts and validity constraints for such lineage records [13]. Distributed tracing provides a complementary operational view: OpenTelemetry represents execution as a graph of spans with identifiers, timestamps, attributes, events, status, and causal links [14]. Recent agent-specific work enriches this substrate. AgentTrace organizes runtime observations into cognitive, operational, and contextual surfaces and connects structured agent logging to telemetry infrastructure [15].

For decision reconstruction, this lineage continues through the authorized actor who commits or modifies a recommendation and the authority grant that permits the action. The resulting record also supports synchronized replay and comparison, rather than serving only as a historical trace.

Business process management separates process design, execution, monitoring, and improvement. Process mining connects these activities by deriving behavioral models from event data and checking observed execution against them [16]. Agentic BPM extends this setting to agents that observe process state and act through organizational tools. Proposed architectures cover the data, reasoning, action, and orchestration needed for such systems [17], while the Agentic BPM manifesto emphasizes bounded autonomy and explainability [18].

Building on process-mining foundations, Park and van der Aalst describe a digital twin of an organization as a processoriented representation connected to operational data and action-oriented analysis [19]. Organization-design research distinguishes digital models, shadows, and twins by the direction and degree of coupling with organizational reality [20]. For agentic decisions, the synchronized projection must retain the state that affected the governed path. DNative-Twin uses isolated execution to test that path before a revised mechanism is released.

Knowledge graphs represent typed entities and relations, while temporal knowledge graphs associate facts with time points or validity intervals. Surveys of knowledge-graph representation and temporal completion describe methods for link prediction, temporal reasoning, extrapolation, and dynamic representation learning [21], [22]. These methods provide useful machinery for evolving multi-relational enterprise data, especially when facts, identities, and dependencies change over time.

Here, typed schema and explicit graph queries turn temporal relations into inspectable checks for missing dependencies, invalid authority, and cross-environment differences. Process models describe flow, guardrails constrain actions, traces record execution, and provenance records lineage. DNative-Twin links these records through a committed decision and makes the trajectory executable under a replay contract.

## VII. DISCUSSION AND LIMITATIONS

DNative-Twin applies when an agentic process has an identifiable decision object and a bounded trajectory whose dependencies, constraints, authority, and outcome can be recorded. These conditions make reconstruction and replay well defined. They can occur in enterprise, personal, clinical, and publicsector settings, under either human review or delegated automation. The framework targets consequential processes that cross a decision boundary; generation without a committed action falls outside this scope.

The instantiation evaluated in this paper uses enterprise processes because their persistent events, approval paths, and outcomes make the required conditions observable. Public logs test graph construction and structural queries. Overlays supply decision fields absent from those logs, and controlled injections isolate selected mechanism failures. Evaluation in another domain requires evidence, authority rules, constraints, and outcomes defined for that domain.

The experiments also show why representation, replay, and verification are separate layers. A graph can localize a represented change, but an unobserved tool state leaves no structural evidence of its consequence. The replay contract records the tool state and expected behavior, while verification results record whether the relevant condition held. These sources complement the graph when the decision depends on information outside the recorded trajectory.

The evidence is bounded by its representation and evaluation settings. Schema design and event-to-graph extraction determine what can be reconstructed, and some queries depend on derived or overlay fields. The BPI 2019 measurements describe the reported sample. Replay stability was measured under one fixed configuration, and the external probes cover two provider-specific models under recorded sampling conditions. The adjudication evidence covers injected benign, material, and unresolved cases, with no held-out critical instance.

The release demonstration evaluates four of the eight declared conditions. A deployment must also evaluate freshness, the complete critical-scenario suite, iteration limits, and replay budget. Because the graph records sensitive operational relationships, deployment also requires access control, retention limits, and privacy-preserving references.

## VIII. CONCLUSION

Reconstructing an agentic decision requires the mechanism that produced and authorized it, not only its final output. DNative-Twin records that mechanism as a trajectory graph and re-executes it under declared conditions. In the enterprise instantiation, the reported mappings construct decision graphs from case-centric and object-centric public logs together with declared overlay fields. The controlled experiments then exercise replay, perturbation, adjudication, and bounded constraint adjustment. In the three-condition injected experiment, unresolved recall increased from 0 to 0.667 with replaycontract state and to 1.0 when injected verification results were available; the held-out set contained no critical instance. On the reported platform, 5,000 BPI 2020 cases were processed in a median 8.889 seconds. The resulting trajectory differences show which state, path, or authorization condition changed before a revised mechanism is released.

## REFERENCES

[1] D. Ferraiolo, R. Kuhn, A. Schnitzer, A. Sandlin, K. Miller, and K. Scarfone, “Guide to attribute based access control (abac) definition and considerations,” National Institute of Standards and Technology, pp. 1– 47, 2014.

[2] X. Gao, B. Xiao, D. Tao, and X. Li, “A survey of graph edit distance,” Pattern Analysis and applications, vol. 13, no. 1, pp. 113–129, 2010.

[3] I. D. Raji, A. Smart, R. N. White, M. Mitchell, T. Gebru, B. Hutchinson, J. Smith-Loud, D. Theron, and P. Barnes, “Closing the ai accountability gap: defining an end-to-end framework for internal algorithmic auditing,” in Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency, ser. FAT\* ’20. New York, NY, USA: Association for Computing Machinery, 2020, p. 33–44. [Online]. Available: https://doi.org/10.1145/3351095.3372873

[4] G. Park and g. U. Leah Tacke, “Procure-to-payment (p2p) objectcentric event log in ocel 2.0 standard,” Oct. 2023. [Online]. Available: https://doi.org/10.5281/zenodo.8412920

[5] A. Berti, I. Koren, J. N. Adams, G. Park, B. Knopp, N. Graves, M. Rafiei, L. Liß, L. T. G. Unterberg, Y. Zhang et al., “Ocel (object-centric event log) 2.0 specification,” arXiv preprint arXiv:2403.01975, 2024.

[6] B. van Dongen, “Bpi challenge 2020: Travel permit data,” 2020. [Online]. Available: https://data.4tu.nl/articles/ /12718178/1

[7] ——, “Bpi challenge 2019,” 2019. [Online]. Available: https://data.4tu.nl/articles/dataset/BPI Challenge 2019/12715853/1

[8] A. A. Hagberg, D. A. Schult, and P. J. Swart, “Exploring network structure, dynamics, and function using networkx,” in Proceedings of the 7th Python in Science Conference, G. Varoquaux, T. Vaught, and J. Millman, Eds., Pasadena, CA USA, 2008, pp. 11 – 15.

[9] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, “ReAct: Synergizing reasoning and acting in language models,” in International Conference on Learning Representations, 2023.

[10] Y. Ruan, H. Dong, A. Wang, S. Pitis, Y. Zhou, J. Ba, Y. Dubois, C. J. Maddison, and T. Hashimoto, “Identifying the risks of LM agents with an LM-emulated sandbox,” 2023, arXiv:2309.15817.

[11] Z. Zhang, S. Cui, Y. Lu, J. Zhou, J. Yang, H. Wang, and M. Huang, “Agent-safetybench: Evaluating the safety of LLM agents,” 2024, arXiv:2412.14470.

[12] National Institute of Standards and Technology, “Artificial intelligence risk management framework: Generative artificial intelligence profile,” National Institute of Standards and Technology, Tech. Rep. NIST AI 600-1, 2024.

[13] L. Moreau and P. Missier, “PROV-DM: The PROV data model,” World Wide Web Consortium, W3C Recommendation, 2013. [Online]. Available: https://www.w3.org/TR/prov-dm/

[14] OpenTelemetry Authors, “Opentelemetry specification: Tracing API and trace semantic conventions,” 2026, accessed: 2026-08-14. [Online]. Available: https://opentelemetry.io/docs/specs/otel/trace/api/

[15] A. AlSayyad, K. Y. Huang, and R. Pal, “Agenttrace: A structured logging framework for agent system observability,” 2026, arXiv:2602.10133.

[16] W. M. P. van der Aalst, A. Adriansyah, A. K. A. de Medeiros et al., “Process mining manifesto,” in Business Process Management Workshops, ser. Lecture Notes in Business Information Processing. Springer, 2012, vol. 99, pp. 169–194.

[17] M. Dumas, F. Milani, and D. Chapela-Campa, “Agentic business process management systems,” in Business Process Management Workshops, ser. Lecture Notes in Business Information Processing. Springer, 2026, vol. 569, pp. 3–14.

[18] D. Calvanese, A. Casciani, G. D. Giacomo, M. Dumas, F. Fournier, T. Kampik, E. L. Malfa, L. Limonad, A. Marrella, A. Metzger, M. Montali et al., “Agentic business process management: A research manifesto,” 2026, arXiv:2603.18916.

[19] G. Park and W. M. P. van der Aalst, “Realizing a digital twin of an organization using action-oriented process mining,” in 2021 3rd International Conference on Process Mining. IEEE, 2021, pp. 104–111.

[20] K. Lyytinen, B. Weber, M. C. Becker, and B. T. Pentland, “Digital twins of organization: Implications for organization design,” Journal of Organization Design, vol. 13, pp. 77–93, 2024.

[21] S. Ji, S. Pan, E. Cambria, P. Marttinen, and P. S. Yu, “A survey on knowledge graphs: Representation, acquisition, and applications,” IEEE Transactions on Neural Networks and Learning Systems, vol. 33, no. 2, pp. 494–514, 2022.

[22] B. Cai, Y. Xiang, L. Gao, H. Zhang, Y. Li, and J. Li, “Temporal knowledge graph completion: A survey,” 2022, arXiv:2201.08236.