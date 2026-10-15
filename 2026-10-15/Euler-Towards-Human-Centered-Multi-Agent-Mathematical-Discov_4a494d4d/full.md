# Euler: Towards Human-Centered Multi-Agent Mathematical Discovery

Qiuyu Chen<sup>1,4</sup>, Qi Wang<sup>2</sup>, Liang Xu<sup>3</sup>, Xin Jin<sup>4</sup> Baao Xie<sup>4,\*</sup>, Wenjun Zeng<sup>4,\*</sup>

<sup>1</sup>Shanghai Jiao Tong University, <sup>2</sup>Institute of Automation, Chinese Academy of Sciences, <sup>3</sup>Zhongguancun Academy, Beijing, China,

<sup>4</sup>Ningbo Key Laboratory of Spatial Intelligence and Digital Derivative,

Ningbo Institute of Digital Twin, Eastern Institute of Technology, Zhejiang, China

Mathematical research drives scientific discovery, and recent advances in AI for mathematics have shown that language-model agents can solve challenging problems and develop substantial mathematical proofs. Yet much of today’s AI-for-mathematics work centers on obtaining a final answer. This emphasis leaves the exploration behind a result dificult for researchers to understand, while long-running investigations face fragmented context and the risk of unsupported claims propagating across reasoning steps. We present Euler, a human-centered mathematical research system built around a shared, structured representation of the research process. Euler organizes mathematical statements, assumptions, proof dependencies, alternative approaches, failed attempts, and verification evidence into connected research graphs. This representation allows researchers to trace how results are developed and supplies agents with persistent, task-relevant knowledge for planning, collaboration, and verification across sessions. Researchers can also select a graph node linked to a saved team state, revisit the agents’ earlier reasoning, and explore alternative directions through continued interaction. We illustrate Euler’s capabilities through three mathematical research case studies and describe the implemented system and its software validation.

Code: github.com/canghaimeng/Euler

October 13, 2026 • Technical Report • Version 0.3.0.dev0

## 1 Introduction

Mathematical research provides the conceptual and theoretical foundations of advances across science and technology. A mathematical result can explain a phenomenon, establish the limits of a method, or open a new line of inquiry. Recent progress in AI for mathematics has substantially expanded the role of language models in this process. Mathematical agents can develop extended proofs, search for counterexamples, use symbolic and formal tools, and coordinate specialized reasoning tasks [1, 2, 4, 17, 18]. These developments suggest a broader role for AI in mathematical discovery and in the day-to-day work of mathematicians.

Much of the current work, however, is organized around the production of a final theorem, proof, or counterexample. Mathematical research also yields valuable knowledge along the way: why a certain approach was attempted, which assumptions turned out to be essential, how partial results fit together, and where a promising line of reasoning failed. Understanding these decisions helps a researcher evaluate a proof, identify its most important ideas, and formulate better questions. This is especially relevant to dificult problems that require sustained exploration and may remain unresolved after substantial efort. An AI research system should make this evolving process accessible so that mathematicians can examine and build on the investigation itself.

Long-horizon reasoning introduces a second, closely related challenge. A research campaign may span many model calls, agent conversations, and tool executions. Relevant definitions and earlier proofs can fall outside the current context window; an agent may repeat an unsuccessful approach or overlook the scope of a previous result. More seriously, an unsupported intermediate claim can be carried into later arguments and gradually acquire the appearance of an established premise. These issues become harder to manage as tasks are divided among agents and resumed across sessions. Reliable continuation therefore depends on retaining the mathematical relationships among prior work, selecting the context relevant to each new task, and checking the evidence used to advance the research.

We develop Euler around a shared, structured representation of mathematical research. The representation records exact statements and assumptions, dependencies among claims, alternative proof strategies, unsuccessful attempts, and the evidence supporting each result. It serves two complementary purposes. For researchers, it provides an inspectable account of how an investigation develops. For agents, it provides persistent mathematical context and explicit targets for planning, collaboration, and verification. Both interact with the same research records as the investigation evolves.

Euler combines this representation with a multi-agent research workflow. Its research graph organizes open problems and proposed approaches, while a separate graph of admitted results records claims accepted through the configured verification process, building on the Fact Graph architecture of Danus [2]. During execution, agents use the current unresolved problems and their dependencies to select tasks, retrieve relevant prior results and failures, and coordinate further reasoning. Intermediate conclusions enter the shared record with their statements, scope, and checking evidence. The system can consequently continue an investigation across context windows while maintaining a traceable basis for later steps.

The same structure supports human understanding and direct participation. Euler’s workbench presents a hierarchical view of the research, from the main problem to individual claims, proof steps, and evidence. A researcher can follow the premises of a conclusion, inspect an abandoned route, or ask agents about a specific mathematical step. Saved team states are associated with research milestones; from a graph node, the researcher can select an associated saved state and resume the corresponding agent conversations in an independent branch. This permits retrospective questioning of crucial ideas and further exploration of alternatives while preserving the original research history.

This report contributes (1) a structured mathematical research state that connects statements, dependencies, strategies, failures, and evidence in a form shared by researchers and agents; (2) a state-guided approach to long-horizon agent reasoning, including context selection, task coordination, and verification-aware updates; and (3) a research workbench that supports proof inspection, focused human questions, and continuation from saved multi-agent states. We examine three mathematical investigations in optimization, graph theory, and geometric group theory to illustrate how the system exposes proof structure and research history. We report reproducible software and integration validation of the implemented workflow and examine the recorded mathematical cases through their source-linked visualizations.

Figure 1 summarizes the research workflow.

## 2 Research Workflow and System Overview

## 2.1 Mathematical context as a persistent research asset

An extended research project contains several forms of information. Accepted claims are reusable premises. Open obligations describe unfinished work. Routes express proposed ways to connect premises to a target. Experimental and computational artifacts support conjectures or rule out specific constructions. Failed proof attempts identify gaps that may recur when the same mechanism is revisited. These records have diferent evidential status, yet their relationships determine the next productive step.

![](images/3e3ea8e7de920f6327712325be0edba9915186140333e5680d061478295d682b.jpg)  
Figure 1: The Euler research loop. A fixed mathematical target and persistent research records determine an active frontier. Agents receive contexts for selected obligations, produce arguments or computational evidence, and return checked outcomes to the shared state. The same underlying objects are presented to the human researcher through the workbench.

Consider two candidate routes for a target theorem T: one reduces T to A ∧ B, and the other to C. A proof of A contributes to the first route, even while B remains open. A checked counterexample to a proposed universal statement B may invalidate that particular route. If a further reduction expresses C in terms of $D \wedge E$ , the active agenda changes, although the number of open obligations may increase. This example motivates a representation in which the progress of an investigation can be assessed through dependencies and checked state changes. Figure 2 displays the corresponding evolution.

The shared representation also supplies a common reference system for human–agent communication. A user question such as “Does the argument for B require the extra regularity assumption?” can be attached to the version of B under discussion. The source statement, its assumptions, and its known evidence can then be included automatically in the resulting task. On completion, the researcher can inspect the answer at the same object.

## 2.2 System requirements

These observations lead to four design requirements. Precise mathematical identity binds statements to assumptions, quantifiers, domain restrictions, and definition versions. Evidence-linked explanations make each accepted conclusion and unresolved dependency inspectable. Actionable research state supports targeted retrieval, scheduling, and updates rather than requiring repeated interpretation of full transcripts. Durable human participation lets researchers review the state, formulate scoped follow-up work, and recover prior research milestones with their team context.

## 2.3 System architecture

Euler is organized around a persistent mathematical research state that is accessible to both researchers and AI agents. The research storage layer maintains private worker notes, project-shared findings, admitted mathematical facts, and the graph of active research goals and approaches. A researchcontrol layer selects mathematical tasks, assembles their relevant context, manages execution budgets, and updates the state when checked results arrive. The agent execution layer connects the lead and its specialized teammates to mathematical reasoning, theorem retrieval, computation, and verification tools. The research workbench reads the same state to display proof dependencies, active exploration, evidence, and team conversations, and to prepare focused follow-up work or branch from a saved team state.

A project begins when the researcher specifies a mathematical target and a team. The lead introduces candidate approaches and assigns work against open goals. Each agent receives a bounded task context assembled from relevant statements, dependencies, earlier attempts, and evidence. Proposed results pass through the corresponding checker and are linked back to the research graph. Throughout the investigation, the workbench exposes the evolving proof structure and allows the researcher to inspect a claim or continue a specific line of reasoning. This architecture makes the research representation the shared interface between mathematical understanding and sustained agent execution, as depicted in Figure 1.

Relationship to Danus. Euler retains the verifier-gated Fact Graph and three-level memory foundation of Danus [2]. Its additional mechanisms organize the unfinished mathematical investigation and connect that representation to execution and human participation. Table 1 summarizes the publicly documented architecture of both systems.

Table 1: Published architecture features of Danus and Euler.
<table><tr><td>Capability</td><td>Danus [2]</td><td>Euler (Ours)</td></tr><tr><td>Verified result memory</td><td>√ Fact Graph</td><td>√ Fact Graph + evidence links</td></tr><tr><td>Open research state</td><td>X Not explicit</td><td>√ Goals, routes, blockers</td></tr><tr><td>State-guided context</td><td>X Not documented</td><td>√Scope-aware dependency retrieval</td></tr><tr><td>Human proof inspection</td><td>X Dashboard / reports</td><td>√ Proof reader + graph views</td></tr><tr><td>Branchable continuation</td><td>X No public team-state recovery</td><td>√ Checkpoints + native session forks</td></tr><tr><td>Agent coverage</td><td>X Codex-centered team</td><td>√Multi-provider adapters</td></tr></table>

## 3 Structured Mathematical Research State

## 3.1 Problem contracts and mathematical scope

Each project begins with a ProblemContract P. It records the target statement $T ,$ its mathematical scope Γ, the target kind, source and definition versions, and the required evidence level. The scope includes applicable assumptions, quantified variables, domains, and object restrictions. A contract defines the proposition against which the investigation is evaluated. Research directions that modify these conditions are recorded under derived contracts with explicit links to their parent problem.

The distinction is important when an intermediate claim uses stronger assumptions. For instance, a proof of $T ( u )$ at $u = 2$ supplies evidence for that parameter value. A statement quantified over an interval requires a separate obligation and corresponding evidence. Similarly, the logical forms ∀u ∃x $P ( u , x )$ and ∃x ∀u $P ( u , x )$ describe diferent research targets. Versioned scope allows the system to retain these distinctions through task generation and verification.

An obligation o stores an exact statement $q _ { o } ,$ , a contract reference, a mathematical scope $\Gamma _ { o } ,$ provenance, an evidence requirement, and its current status. Its identity is derived from stable semantic fields, while operational status is maintained through recorded transitions. This permits an obligation to be reopened after evidence changes without losing the earlier history.

## 3.2 Persistent memory and the two research graphs

Euler uses three memory tiers. Worker-local memory contains private notes and execution records. Shared global memory contains typed findings, including proof attempts, examples, counterexamples, objections, dead ends, and guidance. The Fact Graph $\mathcal { G } _ { t }$ stores statements and proof texts admitted by the configured verification pathway. Facts include predecessor IDs, glossary information, and external references; their content-addressed identifiers support stable lookup and dependency-based revocation. The Fact Graph and its admission gateway are developed from the Danus architecture [2].

The Research Graph $\mathcal { R } _ { t }$ describes the wider investigation. Its objects include obligations, candidate routes, evidence records, and typed relations. A route r identifies a parent goal $q _ { r }$ , a collection of premise obligations Prem(r), and a proposed closing argument. A proof route has the form

$$
\Gamma _ { r } \vdash \left( \bigwedge _ { o \in \mathrm { P r e m } ( r ) } q _ { o } \right) \Longrightarrow q _ { r } .\tag{1}
$$

The closing implication is itself subject to an evidence requirement. Multiple routes attached to one goal represent alternative approaches; the premises within each route form conjunctive requirements. Supported reduction relations, equivalences under stated assumptions, refutations, and descent arguments therefore have explicit mathematical and evidential scope.

The implemented relation vocabulary includes REQUIRES, REDUCES\_TO, EQUIVALENT\_UNDER, SUSPECTED\_SAME, BLOCKED\_BY, REFUTED\_BY, and DESCENDS\_BY. Relations asserting mathematical consequences require supporting evidence. Exploratory relations preserve the status of a proposed connection until that support is supplied.

A blocker identifies an unresolved gap, failed mechanism, or scoped counterexample together with the route or claim it afects. It carries its source and conditions for reconsideration. For example, an obstruction to a candidate construction may rule out that construction under a specified parameter regime. Recording this conclusion with its scope makes the information reusable when a later worker considers a similar route.

## 3.3 The active research frontier

Definition 1 (Active research frontier). For a project contract P and persistent research state at time t, the active frontier is the projection

$$
\mathcal { F } _ { t } = \Phi _ { \mathcal { P } } ( \mathcal { R } _ { t } , \mathcal { G } _ { t } ) = ( O _ { t } , R _ { t } , B _ { t } , E _ { t } ) ,\tag{2}
$$

where $O _ { t }$ contains unresolved obligations requiring attention, $R _ { t }$ contains currently considered routes, $B _ { t }$ contains applicable blockers, and $E _ { t }$ references evidence relevant to those objects. The inherited graph edges preserve their dependency and source relationships.

The frontier concentrates the current agenda without discarding the earlier research record. It can be reconstructed from persistent state after an interruption and can change when an obligation is discharged, a route is revised, a blocker is resolved, or new evidence changes the applicability of an earlier claim. Its structure supports both the workbench’s main-line and frontier views and the controller’s generation of candidate actions.

A newly initialized project may have one root obligation and no proposed routes. The lead agent adds possible reductions and registers their premise obligations. A verified intermediate statement may close an obligation, while an objection to a reduction can return its route for revision. An archived route retains its original argument and supporting artifacts, allowing it to be reopened after a new fact, tool, or assumption changes the situation.

![](images/d848fff353d9b5aadc9fc79fd933e6ee20eb7d853320595f474476eca868ce7f.jpg)  
Figure 2: An illustrative $\mathrm { A N D / O R }$ frontier. Two proposed routes compete to establish T. Evidence closes A and excludes the route requiring $B ;$ a checked reduction of $C$ creates obligations $D$ and E. The active fringe moves as supported claims and route relations change.

## 3.4 Evidence and state transitions

Evidence is represented by typed artifacts associated with exact claims, scopes, contracts, sources, and checker outcomes. The current implementation defines heuristic observations, exact finite results, checked counterexamples, LLM-checked proofs, certificate-checked results, and Lean-kernel-checked proofs. The evidence requirement is assigned when the problem or obligation is created. Admissibility depends on a compatibility relation among evidence kinds and on validity, scope alignment, provenance, and checker outcome.

For an obligation o and a candidate artifact $e ,$ we write the admissibility predicate as

$$
\operatorname { A d m i s s i b l e } ( \epsilon , o ) = \operatorname { V a l i d } ( e ) \wedge \operatorname { S c o p e M a t c h } ( e , o ) \wedge \operatorname { C l a i m M a t c h } ( e , o ) \wedge \operatorname { K i n d M a t c h } ( e , o ) \wedge \operatorname { C h e c k P a s s } ( e ) .\tag{3}
$$

where the required kind determines the acceptable checking procedures. Heuristic observations can inform exploration while obligations requiring a checked result retain their stronger completion condition. The implementation further binds evidence to the contract identity and records the status of dependent objects when source evidence is invalidated.

The gateway supports a separate LLM proof-review service. A verdict from this service establishes that an independent model review accepted the submitted proof according to the configured policy. Formal workflows record Lean kernel checks together with attestations relating formal declarations to the intended mathematical statement. Source-backed computational results are checked against their stated domains and, where needed, their realization as original mathematical objects. These distinctions remain available to the reader as evidence metadata.

A state update is applied after a worker action $a _ { t }$ produces an output $y _ { t }$ and associated checks $z _ { t } \mathrm { : }$

$$
( \mathcal M _ { t + 1 } , p _ { t } ) = \mathrm { R e c o n c i l e } ( \mathcal P , \mathcal M _ { t } , a _ { t } , y _ { t } , z _ { t } ) ,\tag{4}
$$

where $p _ { t }$ is a progress receipt identifying the changed objects and the evidence supporting the change. The controller records obligation discharge, route reduction, scoped refutation, blocker refinement, and operational outcomes through distinct transitions. A final theorem requires an exact root-aligned result and a separate final audit. The dependency cascade is re-evaluated when an accepted supporting fact is revoked.

Table 2: Evidence-mediated changes to the research state.
<table><tr><td>Result</td><td>Recorded support</td><td>Frontier effect</td></tr><tr><td>Accepted claim</td><td>Matching statement, scope, required evidence</td><td>Discharge an obligation and update its dependents</td></tr><tr><td>Validated reduction</td><td>Checked closing argument with dependencies</td><td>Register a supported route and expose remaining premises</td></tr><tr><td>Checked counterexample</td><td>Witness with an aligned target and scope</td><td>Refute the affected local claim or route</td></tr><tr><td>Verified obstruction</td><td>Concrete checker or proof artifact</td><td>Refine the blocker and route options</td></tr><tr><td>New proposal</td><td>Source, author, target, and intended mechanism</td><td>Add an exploratory route or task</td></tr><tr><td>Execution outcome</td><td>Bound lease, context, and execution receipts</td><td>Update runtime status and resource accounting</td></tr></table>

The resulting structure carries two complementary meanings: it is a mathematical record that a researcher can read, and a computational state on which the controller can operate. This shared basis is the central design choice of Euler.

## 4 Frontier-Driven Agent Reasoning

## 4.1 Planning and execution over mathematical objects

The research controller converts the current frontier into concrete work. An action identifies its mathematical target, intended operation, expected evidence, selected agent or tool, and resource allowance. The implemented action kinds include proof construction, counterexample search, literature retrieval, computation, verification, and audit. The controller also tracks whether prerequisites are available, whether an equivalent action has already been executed, and whether the proposed mechanism has encountered a relevant obstruction.

A lead agent proposes mathematical directions and delegates work. Workers carry out scoped assignments and publish intermediate findings to shared memory. The persistent record allows the lead to combine results across workers and across separate sessions. When a project contains alternative suficient conditions, the lead can explore these in parallel. Within a chosen route, independent premise obligations may be assigned separately, while steps whose proof depends on a particular developing argument can be kept within one worker’s context.

At each round the controller uses the current research view and resource policy to admit a feasible subset of candidate actions. The scheduler classifies actions by their relationship to the active proof path, readiness, novelty, and expected use of the available resource budget. It returns explicit reasons for selection, deferral, or rejection. Examples include missing or stale dependencies, duplicate actions, recurrence audits, exhausted slots, and budget constraints. The selected action receives a compiled context and an execution lease before the corresponding provider is launched.

The controller maintains a distinction between a newly recorded claim, its examination, and the resulting change to the mathematical state. Each action completion is reconciled against the contract, source versions, and recorded evidence. This produces an inspectable connection between an agent’s work and the evolution of the research frontier. Algorithm 1 summarizes the operating cycle implemented by the control and execution modules.

Algorithm 1 Frontier-driven research cycle   
Require: Contract P; persistent research state M; resource policy b   
1: Restore the current research view and unsettled action receipts   
2: while the project remains active do   
3: Incorporate completed outputs through evidence-aware reconciliation   
4: Update the frontier $\mathcal { F }  \Phi _ { \mathcal { P } } ( \mathcal { R } , \mathcal { G } )$   
5: if a root-aligned candidate satisfies the prerequisites for final audit then   
6: Run the configured final audit and record the acceptance decision   
7: if the root obligation is accepted then   
8: return accepted checkpoint   
9: end if   
10: end if   
11: Review repeated or blocked routes; update continuation recommendations   
12: Propose mathematical actions against frontier objects   
13: Select a feasible action set under dependency, slot, and budget policies   
14: for each selected action a do   
15: Compile its required mathematical interfaces and bounded historical context   
16: Bind contract, context, allowance, and lease to the worker assignment   
17: Dispatch and persist start/launch receipts   
18: end for   
19: Await completions or save a resumable checkpoint   
20: end while

## 4.2 Dependency-aware context compilation

A research project can accumulate substantially more material than a single model context window can hold. Euler retains the complete record in persistent storage and creates an action-specific ContextPacket for each worker. Its construction is driven by the target obligation and the dependencies needed to reason about that obligation.

Let a be a selected action, $D _ { t } ( a )$ the necessary dependency interfaces, and $H _ { t } ( a )$ an eligible subset of historical findings and blockers. Conceptually, the context compiler implements

$$
\begin{array} { r } { C _ { t } ( \boldsymbol { a } ) = \mathrm { C o m p i l e } ( \mathcal { P } , \boldsymbol { a } , \mathcal { F } _ { t } , D _ { t } ( \boldsymbol { a } ) , H _ { t } ( \boldsymbol { a } ) ) , } \end{array}\tag{5}
$$

$$
\mathrm { s i z e } ( C _ { t } ( a ) ) \leq L _ { a } , \qquad | H _ { t } ( a ) | \leq K _ { a } ,\tag{6}
$$

where $L _ { a }$ and $K _ { a }$ are the configured context and historical-recall bounds. The compiler preserves mandatory statements, assumptions, proof interfaces, and required evidence references. Optional history is included within the remaining allowance. If the mandatory material exceeds the available limit, the controller reports an explicit context failure or deferral.

Dependency recall follows the graph of admitted results and unresolved obligations associated with the action. An accepted theorem can be referenced through its statement, hypotheses, and identifier. The full proof remains retrievable for detailed inspection or auditing. Context construction validates required predecessors against their current versions and validity status. This reduces the need to reproduce a large transitive sequence of proof texts in every worker prompt while retaining access to the derivations on which the task relies.

Historical recall supplies nearby failed attempts, counterexamples, obstacles, and earlier verification objections. Retrieval uses explicit target and branch links followed by bounded lexical search and scope-aware selection. An earlier failed proof of a premise may therefore be useful when a new worker attempts that premise using a modified technique; a counterexample outside the task’s parameter regime is retained as historical information with its original conditions. The context packet includes a manifest of selected and omitted material, allowing execution records to identify what the worker received.

The compiler places prerequisite assumptions and previously admitted theorem interfaces in the worker’s task context while preserving full proofs for inspection. Version checks and the omission manifest make the supplied material explicit and traceable to the persistent research state.

## 4.3 Reusing unsuccessful work and detecting recurrence

Unsuccessful investigations provide information about the next useful experiment. Euler stores typed findings such as attempted reductions, verifier objections, counterexamples, and dead ends, with links to the relevant branch and subgoal. The execution layer can group repeated failure signatures and request a synthesis of the shared obstruction. The resulting summary remains a research record with its associated provenance and scope.

The control layer also examines recurrence within the Research Graph. A route can return to a previously encountered obligation even after its notation or auxiliary definitions have changed. Recurrence analysis compares graph structure and the versions of applicable premises, evidence, and tools. Strongly connected components identify closed loops in the projected dependency structure, and the controller examines whether those loops contain valid descent relations or alternative exits.

For example, an induction argument for $P ( n )$ may reduce the step to $P ( n - 1 )$ through a justified implication and a base case. The dependency relation revisits the same predicate schema while decreasing a well-founded parameter. The research state records such descent evidence explicitly. When repeated actions return to an unchanged obligation without a comparable justification or new evidence, the policy can request an audit, test a counterexample, revise the route, or park the approach until its reopening conditions are met.

Recurrence is therefore evaluated at the level of an afected mathematical route. A new fact unrelated to that route leaves its outstanding blocker unchanged. In contrast, a newly established bridge lemma can change the viability of multiple routes at once. The retained graph supplies the scope and dependency information needed for this distinction.

## 4.4 Verification, computational evidence, and final acceptance

Mathematical evaluation is integrated into execution at several levels. Proposed natural-language proofs pass through the independent LLM verification gateway when their contract permits that evidence kind. Euler also supports checked finite computations and a Lean integration that runs a pinned project and records the kernel-check result, source hashes, selected declarations, and permitted axioms. An alignment attestation binds the formal declaration to the intended natural-language statement. The evidence requirements of the corresponding obligation determine whether a checked artifact is suficient for discharge.

The current built-in compute service includes a bounded integer-polynomial counterexample workflow, with a separate exact checker and an original-object realization review. Agents may invoke additional mathematical tools through their configured environments. Exact computations, incomplete searches, timeouts, and tool failures are recorded with their separate outcome types and scopes. This information is available both to the controller and to the human reader.

At the project level, acceptance involves the original contract. A successful component check closes the corresponding local obligation. A root-level acceptance requires the matching root claim, supporting evidence at the required level, and a distinct final audit. A checked refutation must satisfy the problem’s original assumptions and, when the counterexample is formulated in an auxiliary model, its mapping to the original mathematical object. The control plane records terminal outcomes such as Solved or Refuted separately from resumable operational states such as budget exhaustion, frontier stagnation, missing external input, and infrastructure failure.

## 4.5 Resource accounting and long-horizon continuation

Tasks are admitted against resource policies that account for monetary cost, reported tokens, wall time, compute use, and concurrency. Let $\mathbf { u } _ { t }$ denote settled additive resource use, $\ell _ { t }$ the outstanding reservations, ${ \widehat { \mathbf { c } } } ( a )$ the allowance of action $a ,$ , and b the resource envelope. The additive dimensions satisfy

$$
\mathbf { u } _ { t } + \ell _ { t } + \sum _ { a \in S _ { t } } \widehat { \mathbf { c } } ( a ) \preceq \mathbf { b } ,\tag{7}
$$

for an admitted batch $S _ { t }$ . Deadlines and simultaneous execution slots are checked separately. Reported usage can be incomplete, so missing quantities are preserved as unknown in the accounting record.

The scheduler allocates efort among load-bearing obligations, low-cost discriminating tests, verification, and exploratory routes. Resource reservations also preserve capacity for the final audit. A research run can pause with its graph and pending obligations intact and resume under an updated user allowance. This enables long investigations to retain their mathematical history without requiring an uninterrupted agent session.

## 5 Human Understanding and Research Continuation

## 5.1 Reading a mathematical investigation

The workbench is a visual projection of the shared research state. Its entry point is the original problem contract and the current status of the main goal. The overview presents a compact proof structure and the active routes leading to that goal. The Proof Reader expands individual records into their exact statements, direct premises, arguments, checking artifacts, and later dependent claims. The Relation Map exposes upstream and downstream connections. Live Research presents active assignments and recent investigations; Diagnostics displays execution and control events; Team Interaction provides persistent conversations with registered agents.

The interface supports a progression from overview to mathematical detail. A researcher examining a conclusion can inspect its support recursively and identify where a proof depends on an unproved premise. A researcher reviewing an unsuccessful approach can locate the afected route, read the counterexample or verifier objection, and see how the remaining obligations changed. The same inspection path is available for an open project whose main goal has not yet been settled.

The workbench distinguishes the recorded status of a mathematical object through its evidence type and corresponding source links. The original mathematical text is preserved in the reader, with summaries used for navigation. This is particularly useful in large projects where a small overview module represents several lower-level derivations. Historical case-study records can be presented through a source-bound projection; writable research controls resolve exact current backend identities.

## 5.2 Human questions attached to mathematical objects

A researcher may select a current obligation, route, blocker, evidence record, relation, or contract and open its canonical object inspector. The inspector supplies the source object’s statement, scope, provenance, and version. The researcher then specifies an intended operation and reviews the resulting task before execution. Implemented intents include verification or audit, challenge and counterexample search, assumption exploration, and alternative-route investigation.

The request binds the project, source object, object version, source contract, exact statement and scope, reviewed derived target, and allowed resources. The control service calculates a preview of the proposed investigation; submission rechecks the binding against the current research revision. For a new hypothesis or conclusion, the system creates a derived problem contract and corresponding open obligation linked to the original source. The researcher must explicitly start the prepared focused action. This design provides a stable mathematical referent throughout the exchange and makes the outcome traceable to the initiating question.

A focused investigation follows the same execution path as a scheduled agent task. The control plane compiles its source context, checks worker and budget policy, issues a lease, and dispatches the action. Its returned evidence is evaluated under the derived contract and reconciled with that action. The UI displays the recorded stages of the interaction, including validation, work creation, execution, and outcome. A single focused request authorizes one bounded action; the user can initiate a subsequent investigation after reviewing its result.

The current control implementation supports pause, resume, and stop at the campaign level. During a pause, new ordinary tasks are withheld, and issued or running leases follow the selected drain or cancellation policy. A separately confirmed focused action can run while the ordinary campaign remains paused. Stop also blocks focused starts. Resume checks outstanding settlements, dependencies, and the current resource allowance. These operations are persisted so the human-readable research history includes the circumstances under which work was authorized and resumed.

## 5.3 Recoverable milestones and persistent team sessions

A researcher often needs to revisit an earlier point with the information and team context available at that time. Euler stores milestones in private local Git history. A checkpoint captures project files, research graphs, evidence artifacts, assignments, receipts, and supported native conversation archives. Its manifest records file hashes and omitted runtime material. Capture occurs at a quiescent boundary, so a milestone corresponds to a consistent recorded team state.

Restoring a milestone creates an independent workspace. The original investigation remains intact, while the restored branch contains the captured mathematical graph, source files, tasks, and team metadata. The workbench associates each branch with its source commit and permits researchers to navigate its proof structure. A follow-up question can then be posed against the restored research state. Figure 3 focuses on the main proof dependencies in the Lewis–Overton project.

![](images/f0bb020a3c10ed66572c143e4d4d75bd87e7f81c9597e113429f555c83ec0f14.jpg)  
Figure 3: Selected dependency chain from the Lewis–Overton research graph. The displayed route runs from rate semantics and scale-free update through the five-trial and four-trial constructions to distinct sharp rates and the final initial-state-dependence result.

The native team subsystem extends file recovery to provider sessions. It supports Codex, Claude Code, Gemini CLI, Grok, and Cursor Agent through provider-specific adapters. Depending on the provider, the restored member receives a native session fork or continues a preserved session in a branch-private store. Provider credentials remain outside the milestone archive. The restoration procedure checks availability of the required native-session material before launching a new branch.

The restoration dialog summarizes the selected proof object, the checkpoint to recover, the contents of the saved branch, and whether the members’ native conversations are available for continuation. A researcher can optionally attach a follow-up question before opening the restored workspace. Figure 4 shows this step in the Euclid onboarding exercise; once the workspace is created, the recovered team conversations can be resumed there.

![](images/775e1eea36729be85ffc6031e9e8bf3c0c2bf30a6c4df519c5f0140a8f060435.jpg)  
Figure 4: Milestone-restoration dialog in the Euclid onboarding exercise. The panel identifies the selected proof object, the saved checkpoint, the recovered branch contents, and the availability of native team conversations. The researcher may attach an optional follow-up question before opening the restored workspace.

## 5.4 From a research question to a traceable outcome

The workbench links a researcher’s question to a particular mathematical object and its source state. Figure 5 shows the paused parent campaign in the contemporary BFGS demonstration. The researcher can inspect the existing graph and initiate a separately authorized, focused investigation while ordinary dispatch is paused.

![](images/09eb2a3bd179d3d2b5cd0bd5be55f2bf37276776096098b6aaf6fb433c11795a.jpg)  
Figure 5: Paused parent campaign in the BFGS intervention demonstration. The focused investigation has its own explicit start.

For the proposed follow-up, the researcher reviews the new mathematical statement, parameter scope, and task allowance before confirming the request. Figure 6 shows the reviewed five-trial-cycle question. Its resulting task retains the selected source object’s identity and the relevant mathematical context.

![](images/395f7154a0f76edb8f811f5c830a22a9036f8af46dd7a63935c7609b7cc8e0a1.jpg)  
Figure 6: Review of the derived BFGS question, assumptions, and execution allowance before dispatch.

Once the agent has run, the same request identifier links validation, work creation, dispatch, checking, and reconciliation. The workbench presents these records as the source-linked timeline in Figure 7. The depicted mathematical worker and verifier responses are deterministic fixtures; the underlying task and record transitions use the implemented execution services.

![](images/f2ab8f10e319b70de7d1c9c79fe46ba301bd0d18f78c4f8cd551924efd2638e4.jpg)  
Figure 7: Source-linked lifecycle of a focused BFGS investigation, from the researcher’s question to its recorded outcome.

## 6 System Implementation

## 6.1 Architecture and persistence

The current source snapshot is revision ea505c7 of canghaimeng/Euler, with the package version 0.3.0.dev0. The Python package separates the mathematical data model, research controller, execution and provider interfaces, verification gateway, native team management, and web observability services. The core mathematical storage layer and the Research Control Plane use standard-library data structures and persistence primitives; external framework dependencies are added for provider access, MCP, verification HTTP services, and the workbench.

The euler.core package maintains worker-local notes, typed project-shared memory, and the contentaddressed Fact Graph. The euler.gateway package exposes role-gated mathematical tools, including fact submission, retrieval, memory updates, and theorem search. Workers can request fact admission; the coordinating agent receives planning and retrieval capabilities; the verifier receives restricted read access. The verification service supplies an independent review result to the gateway, which records the verdict and proposed mathematical content in the appropriate persistent stores.

The euler.control package provides the research-state and runtime machinery. Problem contracts, obligations, routes, relations, artifact records, and checking results have explicit schemas. An appendonly SQLite event store records state-changing operations with project, actor, causation, idempotency, and payload identities. Research views and runtime views are reconstructed from these records. Scheduler reservations, execution leases, context manifests, and result receipts link proposed work to the concrete action eventually executed. Reconciliation settles a result against these links and updates the corresponding mathematical object when its evidence meets the required conditions.

The execution and provider subsystems run command-line agents through provider-specific adapters. The native team subsystem retains member identities, roles, messages, sessions, and checkpoints. The browser workbench is served by euler.observability and renders research objects into reading and interaction views. Its front end includes a self-contained reader, graph visualization, team controls, and English/Chinese interface support. Table 3 lists the main implementation components.

Table 3: Principal software components in the current Euler repository.
<table><tr><td>Layer</td><td>Source modules</td><td>Main responsibility</td></tr><tr><td>Mathematical memory</td><td>core, gateway, verify</td><td>Shared findings, Fact Graph, admission and revocation</td></tr><tr><td>Research control</td><td>control</td><td>Contracts, obligations, frontier, scheduling, context, evidence, final</td></tr><tr><td>Agent execution</td><td>execution, providers, strategy</td><td>audit Worker rounds, provider adapters, consultation and execution receipts</td></tr><tr><td>Team recovery</td><td>workspace</td><td>Native sessions, messages, Git milestones and branch isolation</td></tr><tr><td>Research workbench</td><td>observability</td><td>Proof reading, graph navigation, live state, human requests, team conversations</td></tr><tr><td>Research outputs</td><td>write_paper, human_summary</td><td>Source-grounded manuscript and progress-report workflows</td></tr></table>

## 6.2 Runtime identity, access, and recovery

The research controller retains a mathematical object’s content-derived identity separately from its changing runtime status. A human request also includes a versioned binding to the active project instance. This permits exact source resolution, replay-safe submission, and detection of stale task contexts. The same mechanism is used when a restored branch refers to an earlier source object. Publicly displayed historical modules remain source-backed reading objects until they are connected to a valid native control identity.

The workbench is designed for a locally controlled research machine or private host. The current personal-workbench deployment uses server-configured project and worker allowlists with sameorigin checks; browser interaction does not require a separate control token. Filesystem scope and authorization policy are selected by the host configuration. Native team processes use Linux Landlock or macOS Seatbelt for filesystem isolation; their network capabilities follow host policy. Persistent provider authentication remains on the host, outside the exported public observations and milestone source packages. SSH forwarding supports remote access to a local workbench.

Milestone capture records regular project files, ignored research artifacts, the SQLite event database, and supported native conversation archives at an idle boundary. Restoring a milestone validates the captured content before creating a new workspace and preparing each provider’s session according to its supported recovery protocol. Pending work remains suspended until authorized by the researcher. These mechanisms let earlier research states be inspected and continued while preserving branch provenance.

## 6.3 Deployment and demonstration entry points

The repository provides a no-account dashboard demonstration with a fixed elementary proof workflow, an editable personal-research initializer, and commands for configuring agent teams and milestones. A native research workspace specifies its mathematical target, evidence requirement, and member roster, followed by provider preparation and workbench startup. The bundled Euclid exercise demonstrates lead–assistant exchanges and team restoration. A separate human-intervention demonstration exercises object selection, scoped task creation, dispatch, and reconciliation with deterministic model fixtures.

Software tests cover persistence, permission boundaries, schema validation, role-gated tools, context compilation, scheduling, kernel adapters, filesystem isolation, provider protocols, milestone restoration, and browser rendering. These tests form the implementation evidence described in Section 8. The mathematical case-study archives, by contrast, are source-bound records of investigations conducted across earlier workflow revisions; their papers and formalization projects are linked directly from the examples directory.

## 7 Mathematical Case Studies

We examine three mathematical investigations represented in the Euler workbench: the Nikoghosyan Hamiltonicity conjecture, a Lewis–Overton BFGS rate conjecture, and the four-connected version of the Jankiewicz–Norin–Wise legal-system problem. Each case presents a diferent type of research structure. The first concerns closing a theorem under exactly its original hypotheses; the second requires precise interpretation of an asymptotic assertion and comparison of two parameterized constructions; the third combines constructive proof with failed candidate searches and a hereditary condition.

These investigations are documented in accompanying manuscripts, publicly linked Lean projects, and historical research-state visualizations. The archived records originate from earlier research workflows, including human guidance and separate formalization activity. We use the workbench to study the organization and inspectability of that mathematical material. Table 4 summarizes the exact questions and their characteristic research structures.

Table 4: Mathematical scope and representation questions in the three case studies.
<table><tr><td>Case</td><td>Documented mathematical result</td><td>Research-state focus</td></tr><tr><td>Nikoghosyan</td><td>Hamiltonicity for finite, 1-tough,  $( P _ { 4 } \cup P _ { 1 } )$  -free simple graphs on at least assumptions three vertices</td><td>Prerequisite closure and exact theorem</td></tr><tr><td>Lewis-Overton</td><td>Two nonterminating BFGS executions Source conventions, paired with distinct sharp trial-normalized</td><td>constructions, and parameter scope</td></tr><tr><td>Jankiewicz-Norin-Wise</td><td>rates for  $9 / 5 < u < 1 3 / 6$  A 33-vertex witness for the four-connected legal-system obstruction question</td><td>Candidate exclusions, strategy changes, and hereditary subgraph conditions</td></tr></table>

## 7.1 Nikoghosyan: tracing a Hamiltonicity proof to its assumptions

A Hamilton cycle visits each vertex of a finite graph exactly once. A graph is 1-tough if, whenever deletion of a vertex set S disconnects the graph into at least two components, the number of components is at most |S|. Nikoghosyan conjectured that every 1-tough graph with no induced $P _ { 4 } \cup P _ { 1 }$ is Hamiltonian, where $P _ { 4 } \cup P _ { 1 }$ denotes a four-vertex path together with an isolated vertex [9]. Related work studied the corresponding forbidden-subgraph class under stronger toughness assumptions [10].

The companion development establishes the asserted Hamiltonicity for finite simple graphs with at least three vertices under the stated 1-toughness and forbidden-subgraph conditions [11]. Its publicly linked Lean repository contains a theorem producing a spanning cycle from edges of the original graph and a corollary expressed using SimpleGraph.IsHamiltonian. The proof develops the prerequisite factor-existence and minimum-cycle-factor arguments from the theorem’s graph hypotheses. The formal source and archived checking record identify the target declarations, build environment, and their transitive axiom dependencies.

From a research-state perspective, the decisive issue is the closure of the prerequisite chain. A proof may develop a useful minimum-factor argument that assumes the existence of a suitable factor; the full target additionally requires the factor to be constructed from the original hypotheses. Likewise, a structural cut or local-restoration lemma can be used in the final argument only after its own conditions have been established. The Research Graph expresses these as separate obligations. The main theorem’s dependency closure thereby becomes a concrete object for inspection and review.

Figure 8 shows a compact representation of the main proof modules. Factor existence and terminal interfaces support the toughness contradiction, which in turn leads to the Hamilton-cycle claim. This dependency structure makes the prerequisite chain available for inspection from the main goal.

![](images/6eae9c805e2ec7c9cc54d3a57858a07756c7cbb8cfdff4a85a6967284dfac003.jpg)  
Figure 8: Selected proof modules leading to Nikoghosyan’s Hamiltonicity theorem. The overview exposes the principal prerequisite dependencies.

The corresponding proof-reader entry is shown in Figure 9. Selecting a module opens its source passage, premises, and recorded evidence. The historical main-goal display is associated with the archived formalization audit; the native Fact Graph preserves its separate admitted intermediate records.

## 7.2 Lewis–Overton: comparing two exact asymptotic constructions

Lewis and Overton investigated BFGS behavior for nonsmooth objective functions. Their 2008 analysis included a conjecture concerning the trial-normalized convergence rate of nonterminating runs on a tilted absolute-value objective [13]. The conjecture concerns the dependence of a sharp asymptotic rate on the tilt parameter and the initialization, under specified line-search and counting conventions. The related 2013 work provides broader context for nonsmooth quasi-Newton behavior [14].

![](images/08ee603698d0bc3239df5ee746c32a49f0c1f206bcb047cf592ca83fedb927cb.jpg)  
Figure 9: Nikoghosyan’s original target and its recorded acceptance basis in the proof reader.

The archived investigation first required a precise formulation of the target. The source algorithm fixes the Armijo and weak Wolfe conditions, the treatment of trial step lengths and line-search failures, and the normalization applied to the asymptotic rate. Its counterexample search identified periodic patterns whose states return to scaled versions of their starting configurations. Exact rational calculations were used to investigate these cycles, followed by a parameterized argument establishing that two distinct mechanisms persist over an open interval.

For $9 / 5 < u < 1 3 / 6$ , the two constructed execution families have cycle contractions

$$
q _ { A } ( u ) = \frac { 3 u } { 4 ( u + 1 ) ^ { 2 } } , \qquad q _ { B } ( u ) = \frac { u } { ( u + 1 ) ^ { 2 } } , \qquad \rho _ { A } ( u ) = q _ { A } ( u ) ^ { 1 / 5 } , \quad \rho _ { B } ( u ) = q _ { B } ( u ) ^ { 1 / 4 } < \rho _ { A } ( u ) .\tag{8}
$$

The five-trial and four-trial families therefore yield diferent sharp trial-normalized rates for the same fixed parameter. The corresponding manuscript and Lean development provide the detailed line-search calculations and interval conditions [7].

The Research Graph expresses the result through two independent branches. Each branch must establish its recurrence pattern, validity of line-search trials, and exact contraction. The branches meet at the rate comparison needed for the target conclusion. This proof structure is visible in Figure 10. The workbench also retains earlier single-parameter experiments and the later interval-level obligation, allowing a reader to trace the move from candidate patterns to a general mathematical claim. A researcher can formulate a follow-up about extending one periodic mechanism by selecting that branch and specifying a new parameter scope.

![](images/f10a7626762c4da5326167dfc6bf3d74b3e1b90de7c44e45784c3bfa6dabdf61.jpg)  
Figure 10: Proof structure for the Lewis–Overton investigation. The five-trial and four-trial execution families establish distinct sharp rates, which are compared to derive initialization dependence. Each module links to its exact conditions and evidence.

## 7.3 Jankiewicz–Norin–Wise: retaining constructive and negative research evidence

Jankiewicz, Norin, and Wise introduced legal systems as combinatorial certificates relevant to virtual fibering questions for right-angled Coxeter groups [15]. Their Problem 5.2 asks for a graph satisfying global connectivity and curvature requirements while excluding a legal system in a specified family of subgraphs. We consider the four-connected version: a finite four-connected graph with girth at least four and nonnegative Charney–Davis curvature, such that every four-connected ordinary subgraph admits no legal system.

The companion result gives a graph with 33 vertices and 66 edges. It is four-regular and four-connected, has girth four and Charney–Davis curvature one, and satisfies the required legal-system obstruction [8]. The formal development records the global graph properties and the ordinary-subgraph statement. Its proof combines structural arguments with finite certificates, some of which are checked through Lean’s native\_decide procedure. The stronger variant involving all three-connected ordinary subgraphs remains a separate question.

The problem illustrates why unsuccessful candidate tests should remain part of the research state. Early exploration considered natural regular, connected, triangle-free graphs. Computation produced legal systems for several proposed witnesses. A positive legal-system certificate ruled out its particular candidate as a solution to the negative-witness task. These findings motivated a change from screening named graphs to constructing a host that preserves a known obstruction under a suitable modification. The later solution combines a seed obstruction, a four-cycle restriction argument, and a three-cap construction.

Figure 11(a) summarizes two load-bearing requirements: establishing the host’s global properties and showing the hereditary legal-system obstruction for its relevant subgraphs. Panel (b) presents a parked candidate-search route next to the completed constructive route. The record of an 11-vertex cap with a legal system also clarifies why this particular host does not satisfy the stronger three-connected requirement. These separate routes let a researcher inspect the outcome of each strategy and the mathematical scope of its evidence within one investigation.

## 7.4 What the case studies show

Across the three investigations, the workbench exposes diferent aspects of the mathematical process. The Nikoghosyan graph makes prerequisite completeness inspectable; Lewis–Overton retains the detailed semantics and competing constructions behind a rate comparison; Jankiewicz–Norin–Wise connects failed candidates to a change of construction strategy. Each case also provides an entry point for further inquiry: audit a prerequisite, vary a parameter condition, or revisit a parked route. The companion Lean repositories and manuscripts provide the mathematical artifacts for each case. The visualizations preserve source-linked records of investigations conducted across earlier workflow versions, including accepted dependencies, unresolved steps, and changes of approach.

## 8 Evaluation

This section reports completed software and integration validation and describes the mathematical research records available for inspection in the three case-study archives. These results document the implemented workflow, its evidence boundaries, and the source material presented by the workbench.

## 8.1 Software and integration validation

At commit ea505c7, the repository’s GitHub Actions workflow completed its ofline tests on Python 3.10 and Python 3.13 with 1,352 passing tests and two skips in each environment. The nativeteam workflow completed 56 tests on Linux and 53 passes with three skips on macOS. These runs covered protocol adapters and real operating-system isolation while using synthetic CLI sessions rather than paid model calls. A separate release workflow built the wheel and exercised the packaged dashboard in a Chromium browser, including all six workbench views, graph selection, language switching, an empty project, responsive layout, and project isolation checks. The CI configurations and current results are available through the project’s Actions history.

![](images/7019f99fc8b098bfd19f188f5516cf1cc592850a76735e4df42acf767bc43701.jpg)  
(a) Global host properties and ordinary-subgraph obstruction.

![](images/8ec8353cf9bd6f57a86b14a4c9de30fa0956ec3008563e6fe9bb5ad16510b572.jpg)  
(b) Retained unsuccessful search and completed construction route.  
Figure 11: The Jankiewicz–Norin–Wise case. The overview identifies the proof obligations of the constructed host, while the research-route records preserve the candidate-search history and the later successful construction mechanism.

Tests within the research-control package cover mathematical object schemas, scoped evidence, stable identities, event replay, resource reservations, scheduling decisions, context compilation, reconciliation, recurrence, formal-kernel adapter behavior, and final acceptance. Workspace and provider tests cover session preparation, checkpoint capture, restored branch identity, preservation of private files, and supported operating-system isolation. The use of model and checker fixtures permits deterministic examination of protocol behavior, including retry, stale-source, and failure cases.

The human-control demonstration exercises selection of a canonical object, preview and confirmation, one focused task, and return of an evidence-tagged outcome. Deterministic worker and verifier responses travel through the implemented HTTP, dispatch, lease, gateway, and reconciliation services. The demo retains the original parent objective while recording each derived investigation. A sourcelinked timeline makes the sequence inspectable in the browser (Figure 7). A separate two-provider restoration exercise used Codex and Grok on the infinitude-of-primes exercise. That exercise tested native conversation forking, checkpoint file consistency, and renewed lead–assistant exchange in the restored workspace.

These results verify the software’s state transitions, provider protocol handling, and workbench behavior under the reported test configurations.

Table 5: Validation artifacts associated with the current repository snapshot.
<table><tr><td>Component</td><td>Validation coverage</td><td>Observed CI outcome</td></tr><tr><td>Offline Python suite</td><td>Research state, scheduler, verification adapters, recovery, API, and evaluation utilities</td><td>1,352 pass; 2 skip (both Python</td></tr><tr><td>Native team protocols</td><td>Provider session doubles and filesystem isolation on Linux/macOS</td><td>versions) 56 pass (Linux); 53 pass, 3 skip (macOS)</td></tr><tr><td>Packaged dashboard</td><td>Wheel installation, six views, source graph, i18n, small-screen layout</td><td>Browser job passed</td></tr><tr><td>Mathematical examples</td><td>Manuscripts, formal repositories, historical graph snapshots</td><td>Companion artifacts; see Section 7</td></tr></table>

## 8.2 Mathematical trace inspection

The three case-study archives contain inspectable, source-linked research records. The Nikoghosyan view exposes the prerequisite structure behind the Hamiltonicity statement; the Lewis–Overton view preserves the two execution branches and their sharp-rate comparison; the Jankiewicz–Norin–Wise view records the construction route alongside earlier candidate exclusions. These examples are presented in Section 7.

The historical views retain original mathematical passages, graph relations, and available checking provenance. Their manuscripts and formalization repositories provide the corresponding mathematical artifacts.

## 9 Related Work

Mathematical reasoning agents. Danus combines multiple reasoning agents with Fact-Graph memory and verification to support sustained mathematical exploration [2]. Euler builds on that admission and memory foundation and extends the research-control and human-workbench layers. Rethlas/Archon investigates automated conjecture resolution with formal checking [1]; MMAT coordinates mathematical agents through control and execution roles [4]. Albilich explores steerable proof-state orchestration and computer algebra, while Magenta and LEAP develop workflows with formal verification feedback [16, 17, 18]. These systems provide complementary designs for orchestrating extended mathematical work.

Human–AI mathematical collaboration. AI Co-Mathematician studies a workbench organized around mathematicians’ participation, progressive reading of research artifacts, and steering of investigations [3]. Human feedback and intervention also appear in multi-agent research systems such as Danus and MMAT [2, 4]. Euler places a canonical mathematical object at the center of this interaction: the object read by the researcher is the same object referenced by the scheduler, context compiler, and reconciled outcome. Milestones extend this continuity across saved agent-team states.

Scientific exploration and persistent context. AI Co-Scientist and AI Scientist-v2 organize iterative scientific exploration through hypothesis management, feedback, and search [19, 20]. Work on long-context access and agentic context engineering motivates structured approaches to recalling relevant information from large histories [5, 6]. Euler specializes these ideas to mathematical premises, obligations, counterexamples, and proof dependencies, using the current research frontier to determine which historical material enters each agent task.

Neural and symbolic representations. Neural–symbolic research considers ways of combining learned methods with explicitly represented knowledge and constraints [21, 22]. The symbolic structures in Euler specify research objects, their identities, relations, and permissible evidencesupported transitions. Language models provide proof strategies and mathematical arguments within this persistent scafold, while theorem checkers and exact computations supply domain-specific validation.

## 10 Conclusion and Future Directions

The central design principle of Euler is that mathematical research should remain intelligible and actionable throughout an extended investigation. A shared representation allows the system to coordinate agent work while giving researchers direct access to the mathematical structures underlying that work. Exact statements and scopes anchor the investigation; proof routes and blockers expose its developing logic; evidence records explain which conclusions have been admitted; and milestones preserve its history. The active frontier connects these records to the next actions the system can perform. The implemented workbench makes the recorded proof dependencies and research decisions inspectable, while the control plane connects focused human questions and agent actions to their context, verification evidence, and resulting updates. The three mathematical cases illustrate these capabilities in concrete investigations.

Several research questions remain. The usefulness of the representation depends on the quality of the proposed routes, the accuracy of scope assignments, and the ability to retrieve the relevant dependencies and failures. LLM proof review has a diferent assurance level from Lean-kernel verification, and formal statements require alignment with their intended mathematical meaning. Applying the current architecture to a wider range of mathematical fields and problem classes will help determine its generality, reliability, and practical value for sustained reasoning and human–AI mathematical collaboration.

## Acknowledgements

Euler’s graph design draws on Danus, and we consulted the Rethlas implementation during development. We thank the authors and contributors of both projects for sharing their work.

We thank Professor Michael Overton and Professor Adrian Lewis for the time and care they devoted to reviewing our argument concerning the Lewis–Overton conjecture.

We also thank the mathematicians and research communities whose prior results, definitions, and lines of reasoning made this work possible. Euler’s use of mathematical knowledge and proof structure depends fundamentally on human insight.

## References

[1] Haocheng Ju, Guoxiong Gao, Jiedong Jiang, Bin Wu, Zeming Sun, Shurui Liu, Leheng Chen, Yutong Wang, Yuefeng Wang, Zichen Wang, Wanyi He, Peihao Wu, Liang Xiao, Ruochuan Liu, Bryan Dai, and Bin Dong. Automated Conjecture Resolution with Formal Verification. arXiv:2604.03789, 2026.

[2] Jihao Liu, Guoxiong Gao, Zeming Sun, Bin Wu, Shurui Liu, Jiedong Jiang, Haocheng Ju, Leheng Chen, Ronnie Cheng, Xiping Zhang, and Bin Dong. Danus: Orchestrating Mathematical Reasoning Agents with Fact-Graph Memory. arXiv:2607.06447, 2026.

[3] Daniel Zheng, Ingrid von Glehn, Yori Zwols, Iuliya Beloshapka, Lars Buesing, Daniel M. Roy, Martin Wattenberg, Bogdan Georgiev, Tatiana Schmidt, Andrew Cowie, et al. AI Co-Mathematician: Accelerating Mathematicians with Agentic AI. arXiv:2605.06651, 2026.

[4] Yichuan Cao, Ruichen Qiu, Junqi Liu, Jiaqi Wang, Dakai Guo, Ruyong Feng, Lihong Zhi, and Xiao-Shan Gao. MechMath Agent Team: LLM Driven Agents for Mathematical Research. arXiv:2607.04394, 2026.

[5] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173, 2024. doi:10.1162/tacl\_a\_00638.

[6] Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, and Kunle Olukotun. Agentic context engineering: Evolving contexts for self-improving language models. arXiv:2510.04618, 2025; revised 2026.

[7] Qiuyu Chen. Initialization-dependent BFGS trial rates for tilted absolute values. arXiv:2609.09872, 2026.

[8] Qiuyu Chen. A four-connected graph without a legal system. arXiv:2609.13419, 2026.

[9] Zh. G. Nikoghosyan. Disconnected forbidden subgraphs, toughness, and Hamilton cycles. ISRN Combinatorics, 2013, Article 673971. doi:10.1155/2013/673971. Preprint: arXiv:1207.5132.

[10] Songling Shan. Hamiltonian cycles in tough $( P _ { 4 } \cup P _ { 1 } )$ -free graphs. arXiv:2504.08936v2, 2025.

[11] Qiuyu Chen. Nikoghosyan’s conjecture: Lean 4 formalization. Software artifact, 2026; revision 69f3a96. https://github.com/canghaimeng/Nikoghosyan-conjecture.

[12] Qiuyu Chen. Euler: Towards Human-Centered Multi-Agent Mathematical Discovery. Software snapshot 0.3.0.dev0, revision ea505c7, October 13, 2026. https://github.com/canghaimeng/Euler.

[13] Adrian S. Lewis and Michael L. Overton. Nonsmooth optimization via BFGS. Optimization Online technical report, 2008. https://optimization-online.org/2008/12/2172/.

[14] Adrian S. Lewis and Michael L. Overton. Nonsmooth optimization via quasi-Newton methods. Mathematical Programming, 141:135–163, 2013. doi:10.1007/s10107-012-0514-2.

[15] Kasia Jankiewicz, Sergey Norin, and Daniel T. Wise. Virtually fibering right-angled Coxeter groups. Journal of the Institute of Mathematics of Jussieu, 20(3):957–987, 2021. doi:10.1017/S1474748019000422.

[16] Ting Gong, Michael Ruofan Zeng, and Yong Yang. Albilich: Steerable Proof-State Orchestration for LLM-Based Mathematical Research with CAS Integration. arXiv:2607.27705, 2026.

[17] Joshua Ong Jun Leang, Haonan Li, Zheng Zhao, Xinyi Shang, Wenda Li, Zhengzhong Liu, Erix Xing, Shay Cohen, and Eleonora Giunchiglia. Magenta: Closing the Loop Between Mathematical Reasoning and Lean Verification. arXiv:2609.11319, 2026.

[18] Po-Nien Kung, Linfeng Song, Dawsen Hwang, Jinsung Yoon, Chun-Liang Li, Simone Severini, Mirek Olšák, Edward Lockhart, Quoc V. Le, Burak Gokturk, Thang Luong, Tomas Pfister, and Nanyun Peng. LEAP: Supercharging LLMs for Formal Mathematics with Agentic Frameworks. arXiv:2606.03303, 2026.

[19] Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, et al. Towards an AI co-scientist. arXiv:2502.18864v1, 2025.

[20] Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jef Clune, and David Ha. The AI Scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv:2504.08066, 2025.

[21] Xiao-Wen Yang, Jie-Jing Shao, Lan-Zhe Guo, Bo-Wen Zhang, Zhi Zhou, Lin-Han Jia, Wang-Zhou Dai, and Yu-Feng Li. Neuro-Symbolic Artificial Intelligence: Towards Improving the Reasoning Abilities of Large Language Models. In Proceedings of IJCAI, Survey Track, pages 10770–10778, 2025. https://doi.org/10.24963/ijcai.2025/1195.

[22] Wenguan Wang, Yi Yang, and Fei Wu. Towards Data-And Knowledge-Driven AI: A Survey on Neuro-Symbolic Computing. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(2):878–899, 2025. https://doi.org/10.1109/TPAMI.2024.3483273.

## A Implementation and Record Interfaces

An event in the research-control store includes the project identifier, actor, event type, idempotency key, timestamp, payload hash, and optional causation and correlation identifiers. The control plane replays these events to recover contract, obligation, route, budget, and execution states. Immutable context and execution receipts connect the selected action to its worker result

A focused human request additionally identifies the source object and reviewed target. Representative fields include object\_type, object\_id, object\_version, statement\_hash, scope\_hash, contract\_id, intent, user\_question, requested\_statement, requested\_scope, and resource-policy information. The current object and research revision are checked when the request is admitted and again when its action starts. The response timeline records the linked derived objects, task lease, evidence, and reconciliation outcome.

## B Reproducibility and Artifact Index

The evaluated Euler source is version 0.3.0.dev0, revision ea505c7 (October 13, 2026). Code, test workflows, and documentation are available at https://github.com/canghaimeng/Euler. The mathematical case materials are under examples/nikoghosyan, examples/lewis-overton, and examples/jnw, with linked paper PDFs and Lean developments. docs/DASHBOARD-GUIDE.md and docs/TEAM-RECOVERY.md describe the personal research workflow.