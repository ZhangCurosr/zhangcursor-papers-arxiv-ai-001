# AGENTEVAL: EVIDENCE-BOUNDED ASSESSMENT OF STATEFUL AGENTS

Ruiqi Gao Alibaba Cloud, Alibaba Group

## ABSTRACT

Task completion alone does not determine whether a stateful agent satisfied its obligations, and a recorded judgment does not guarantee that its supporting evidence remains valid. We present AGENTEVAL, a system that carries a contract-bound assessment record through tool execution, state and event capture, report qualification, and case maintenance. It separates what the agent did, what the evidence establishes, and whether the conclusion is supported for present use. On 32 tool-world administrations, independent checks agree on all 30 complete episodes, distinguishing satisfied, violated, and unresolved obligations; two interrupted administrations are retained. Controlled interventions on 48 evidence views expose 36 binding faults missed by hash-only checks without changing agent behavior. Scenario Studio connects reviewed cases to 12 two-turn reports with 24 independently checked answers. A separate public-benchmark study on 192 held-out executions reveals ten completed return subgoals within broader task failures and isolates the effect of missing evidence. These studies show how an executable assessment record makes scope, evidential support, and report reuse inspectable across an agent evaluation workflow.

## 1 INTRODUCTION

A refund agent reports success. Did money move exactly once? Did it modify only the authorized order? If ownership was ambiguous, did it obtain a trusted answer before acting? And can the assessment still be used after its world or evidence changes? Each question concerns the meaning of the result. Stateful agents make this distinction consequential: the same final answer can accompany different persistent effects, and a plausible completion message cannot establish those effects.

Interactive benchmarks make agent behavior measurable through executable tasks and environments (Yao et al., 2024; Liu et al., 2024; Zhou et al., 2024; Jimenez et al., 2024). Policy-aware, procedureaware, and reliability-oriented evaluations already go beyond task completion (Debenedetti et al., 2024; Levy et al., 2024; Cao et al., 2026; Gupta, 2026). Recent work asks whether retained evidence supports a score and whether assessment remains valid as agents and benchmarks change (Gao & Zhou, 2026; Sediqin et al., 2026; Li, 2026; Luo et al., 2026). The practical challenge is to carry these distinctions through task construction, real execution, evidence capture, assessment, report reuse, and subsequent case maintenance.

We frame this challenge as evidence-bounded assessment. An execution has a fixed history, while its admissibility for present use depends on the evidence and contract that still support it. The key design choice is to carry one bound assessment record across execution, report inspection and case maintenance: each stage depends on the same declared obligations and supporting artifacts. An assessment answers three linked questions: What happened? Did it satisfy the declared obligations? What conclusion is supported now? A declared objective names the quantity being assessed; execution conditions specify authorization, interaction, faults, and budgets. Evidence supplies the state and events needed to decide. Version and provenance bindings delimit where a retained conclusion applies.

AGENTEVAL implements this view through a trusted host for permission, transaction and clarification worlds, evaluator-only evidence capture, and content-addressed assessments. Report inspection rechecks support while preserving history. The web workbench connects these paths to source-bound case maintenance and Scenario Studio: business intent, reviewed revisions, multi-turn execution and failure-informed planning.

![](images/41778637ae8a64f573365b3f692ffe82c4d86e9119e64badc139463f359e663c.jpg)  
Figure 1: A completion becomes an assessment through evidence. This illustrative workflow checks a refund against persistent effects and declared obligations. Bound state, event and version records support the judgment and its current qualification. Governed cases connect reviewed admission, actual consumption and withdrawal while retaining historical evidence.

Our contributions are (1) a contract-bound assessment record and trusted execution host, linking goal and process obligations to tool actions and evaluator evidence; (2) report qualification that preserves historical judgments, exposing when changes in evidence or measurement bindings prevent present use; and (3) provenance-bound case maintenance integrated with an operator workbench, connecting source eligibility, reviewed revisions, and versioned use. Tool-world executions and evidence interventions test assessment and qualification; reviewed multi-turn cases test the operator workflow. A separate public-benchmark study examines objective scope and evidence loss.

## 2 RELATED WORK

Interactive tasks and operational evaluation. AgentBench, WebArena, SWE-bench and τ-bench supply executable tasks (Liu et al., 2024; Zhou et al., 2024; Jimenez et al., 2024; Yao et al., 2024). AgentDojo, ST-WebAgentBench and procedure-aware evaluation assess utility, attacks, policy and interaction constraints (Debenedetti et al., 2024; Levy et al., 2024; Cao et al., 2026). Reliability and behavioral testing motivate repeated, condition-specific measurement (Gupta, 2026; Rabanser et al., 2026; Ribeiro et al., 2020; Zheng et al., 2026). AGENTEVAL retains distinct full-task and declared-subgoal assessments.

Evidence and evaluator validity. Judge calibration and trajectory evaluation examine the assessor itself (Zheng et al., 2023; Lù et al., 2025); active testing addresses evaluation efficiency (Kossen et al., 2021). Evidence-supported bounds add a reporting layer to existing benchmarks (Gao & Zhou, 2026), while the Trust Layer audits saved computations, completion claims and repeat stability (Sediqin et al., 2026). Claim-dependent reliability and transported judge calibration further delimit score interpretation (Hardy et al., 2026; Li, 2026). AGENTEVAL binds the assessment contract before execution and carries its identity through current report inspection and versioned case admission. Our contribution is this executable connection; three-valued labels and finite bounds are established tools within it.

Evaluation workflows and maintained suites. An earlier AgentEval uses LLM agents to generate and verify task-utility criteria (Arabzadeh et al., 2024). AgentSuite audits benchmark components; Inspect and Promptfoo provide evaluation workflows (Suh et al., 2026; UK AI Security Institute, n.d.; Promptfoo, n.d.); practical evaluation and generated tests emphasize harnesses and failure traces (Anthropic, 2026; Gupta et al., 2025). Benchmark maintenance addresses evolving coverage (Luo et al., 2026). AGENTEVAL connects stateful execution, evidence requalification and provenancebound admission, measuring these paths separately from task-generation gains.

## 3 ASSESSMENT OBJECTS AND SEMANTICS

An agent configuration A comprises its model, prompts, orchestration, memory, tools, and stopping rules. A condition ${ \boldsymbol { c } } = ( { \boldsymbol { w } } , { \mathcal { O } } , { \boldsymbol { u } } , { \boldsymbol { q } } , { \boldsymbol { f } } , { \boldsymbol { b } } )$ specifies a world, observation interface, authorization, trusted requester context, declared faults, and execution budget. Execution yields an interaction trajectory $\tau = ( o _ { 0 } , a _ { 0 } , r _ { 0 } , . . . , o _ { T } )$ and a full episode ${ \boldsymbol \xi } = ( \tau , s _ { 0 } , s _ { T } , { \mathcal { E } } )$ containing initial/final world state and events. Evaluator evidence e is a retained view of $\xi ;$ some facts are privileged and never enter the target observation interface.

Declared obligations. A task contract Γ names the goal and applicable permission, side-effect, transaction, and interaction requirements. When necessary context defines every obligation as a binary predicate, its nonempty obligation set $J$ gives

$$
Y _ { \Gamma } ^ { \ast } ( \xi ) = \bigwedge _ { j \in J } g _ { j } ( \xi ) , \qquad Y _ { G } ^ { \ast } ( \xi ) = \bigwedge _ { j \in J _ { G } } g _ { j } ( \xi ) , \quad \emptyset \neq J _ { G } \subseteq J .\tag{1}
$$

The subset relation requires the same goal definition and fully specified context. In the transaction world, the full refund contract additionally checks amount, settlement, declared faults and world invariants. Clarification has a conditional, partially defined contract: a bound trusted reply determines the document obligation; a request without that reply can remain unresolved. Its unconditional final-document goal is a different estimand, not necessarily a subset of this partial contract. The historical retail study checks three defined return obligations within requests that can contain other actions. Each contract names its own scope.

Assessment record. We represent a retained assessment as

$$
{ \mathcal { R } } = ( A , \Gamma , c , v , e , { \widehat { Y } } , m , h ) ,\tag{2}
$$

where v binds target, task, world and checker identities; $\widehat { Y }$ contains check results; m records measurement status; and h records provenance and content digests. The record retains what was judged at execution and whether its support currently qualifies for use.

Three separate questions. For a fully defined contract, behavioral truth $Y _ { \Gamma } ^ { * }$ concerns what the agent did; evidence adequacy concerns whether e determines it. A partial contract can additionally lack context needed to define an obligation. Current qualification $Q ( \mathcal { R } , v _ { \mathrm { n o w } } )$ concerns whether retained bindings support the present report. Evidence loss cannot change past behavior; invalid bindings cannot be repaired by relabeling a historical PASS.

## 4 EVIDENCE-BOUNDED ASSESSMENT IN AGENTEVAL

## 4.1 FROM OBSERVATIONS TO EXECUTABLE CHECKS

A versioned specification declares the facts and obligations required by each check. The canonical verification registry retains PASS, FAIL, PARTIAL, UNKNOWN, and INVALID, with check-level reasons. The current tool-world host additionally uses trusted state and event predicates. We preserve native records rather than impose one undocumented status conversion on every entry point.

For a well-bound, fully defined conjunction, a three-valued assessment is

$$
\widehat { Y } _ { \Gamma } ( e ) = \left\{ \begin{array} { l l } { + , } & { \mathrm { e v e r y ~ r e q u i r e d ~ o b l i g a t i o n ~ i s ~ e s t a b l i s h e d , } } \\ { - , } & { \mathrm { a ~ r e q u i r e d ~ v i o l a t i o n ~ i s ~ e s t a b l i s h e d , } } \\ { \gamma , } & { \mathrm { a v a i l a b l e ~ f a c t s ~ e s t a b l i s h ~ n e i t h e r ~ c o n c l u s i o n . } } \end{array} \right.\tag{3}
$$

Identity or contract mismatch is separately invalid; execution interruption is a measurement failure. An established violation need not disappear when an unrelated fact is missing. Conditional checks additionally preserve UNKNOWN when necessary requester context is absent; this does not certify that an uncreated document satisfied a binary goal. Current contracts appear in Appendix $\mathrm { G } ;$ the historical adapter’s distinct composition is retained in Appendix B.

Table 1: Different records permit different uses. Current qualification is checked independently of behavioral outcome.
<table><tr><td>Behavior</td><td>Current support</td><td>Permitted interpretation</td></tr><tr><td>PASS</td><td>Qualified</td><td>Report established contract satisfaction.</td></tr><tr><td>FAIL</td><td>Qualified</td><td>Report a violation; separately check case-source admission.</td></tr><tr><td>UNKNOWN</td><td>Qualified</td><td>Retain unresolved obligations and their declared reasons.</td></tr><tr><td>Any recorded status</td><td>Refused</td><td>Retain the historical record; withhold current report use.</td></tr></table>

Trusted execution and evidence isolation. The host freezes task and measurement identities before running a registered target. The agent receives its task, projected observations, and tool schemas. World actions enforce permissions and transactions. A trusted responder binds clarification replies to a request token and task identity. Initial and final state, action events, model requests, and receipts form evaluator evidence. A content-addressed store binds these artifacts to run, subject, and specification digests. Expected answers, privileged state, and the independent checker remain outside the target request envelope.

## 4.2 CURRENT QUALIFICATION WITHOUT REWRITING HISTORY

Report inspection reopens retained evidence, specification and declared measurement identity. Abstracting the implemented checks,

$$
Q ( \mathcal R , v _ { \mathrm { n o w } } ) = I _ { \mathrm { b i n d i n g } } \wedge I _ { \mathrm { i n t e g r i t y } } \wedge I _ { \mathrm { c o n t r a c t } } \wedge I _ { \mathrm { d e p e n d e n c y } } .\tag{4}
$$

Qualification requires matching run, subject and specification bindings, intact evidence bytes, a supported contract and the declared measurement dependency identity. It permits a current report, including unresolved outcomes, without resolving every obligation. Recorded execution and diagnoses remain unchanged; historical readback depends on artifact availability. Changes outside the declared measurement closure, such as presentation changes, need not invalidate an assessment. The checks do not establish arbitrary program or provider equivalence.

Unresolved-status envelope. For $N = n _ { + } + n _ { - } + n _ { ? } > 0$ completed, independently admissible records, assigning unresolved statuses to zero or one gives

$$
\left[ \frac { n _ { + } } { N } , \frac { n _ { + } + n _ { ? } } { N } \right] .\tag{5}
$$

This is a label-accounting envelope. If latent binary outcomes are defined and established labels are sound, it is also a finite identification bound on satisfaction. Missing contract context alone does not establish those premises. It is neither a confidence interval nor a new estimator. Invalid bindings and interrupted administrations retain separate counts and a declared denominator policy.

## 4.3 GOVERNED CASE MAINTENANCE

A versioned asset $z = ( \Gamma , c , \mathrm { s o u r c e } , v _ { z } )$ has an explicit lifecycle state. Source failure supports a proposal, not automatic correctness. Calibration uses family-specific positive, defect, and, where applicable, missingness controls. Admission records the approved revision and conditions. New consumption checks approval before registering a real run. Withdrawal prevents further use while retaining earlier evidence and consumption receipts.

Current worlds expose closed recipe families with bounded permission, amount, and documentcontent parameters. Parameter variation extends coverage within those families; it does not establish new task semantics or independent generalization. Historical judgments retain their original contracts throughout transitions. Case maintenance concerns evaluation assets; the evaluated agent remains a separate target.

![](images/e02bd2c6eea724ce7e0cbedbb60ba95eef1846650c3412ddc04700c2f74452e3.jpg)  
Figure 2: Three connected product workflows. Scenario Studio constructs reviewed multi-turn cases. Tool-world evaluation separates agent-visible observations from privileged state and event evidence, then qualifies the retained report. Case governance admits and consumes assets in new runs; withdrawal preserves prior records. The paths share a registered execution host and evidence infrastructure, with distinct assessment contracts.

## 4.4 CONDITION-INDEXED REPORTS

For condition c, a report names the evaluated objective, opportunity set, task components, versions, and supporting evidence. Decided-only satisfaction and coverage are

$$
\widehat { p } _ { \mathrm { d e c } } = \frac { n _ { + } } { n _ { + } + n _ { - } } , \qquad \kappa = \frac { n _ { + } + n _ { - } } { N } .\tag{6}
$$

The first quantity is undefined without decisions and depends on which opportunities remain decidable.   
Under a partial contract it is a decided-label ratio, rather than an unconditional binary success rate.   
Reports retain unresolved reasons and qualification diagnoses alongside definite outcomes.

Comparisons preserve the intended relation between conditions. An equivalent wording transformation should retain goals and obligations; a permission or fault change can alter legitimate behavior. An evidence intervention freezes behavior and changes only the assessor’s view. The historical study exercises wording and evidence contrasts; the current worlds expose controlled authorization, transaction-fault, and trusted-reply conditions. Repetitions, parameter variants, and evidence copies remain distinct from independent task structures.

A software claim ceiling further constrains report interpretation using target access, observability, and evidence. Internal causal explanations require more than outcome records, and comparisons require compatible evaluation envelopes. These mechanisms make scope explicit without replacing the assumptions needed for scientific inference.

## 5 SYSTEM AND OPERATOR WORKFLOW

AGENTEVAL organizes evaluation into three connected workflows: reviewed multi-turn cases, stateful tool execution, and governed case maintenance (Figure 2). They share an execution host and evidence infrastructure while retaining their own task contracts. Appendix G details the interfaces and obligations. The public-benchmark study uses a separate retail adapter; its protocol and results are reported independently.

Real tool execution. The execution host supports Bailian model decisions through world tool actions and Qoder through a constrained native MCP session. Both connect to permission, transaction and clarification environments. Target registration binds the configuration before execution; admission checks, request budgets, deadlines, cancellation and recovery govern each run. The host captures actions and evidence and produces a durable report. Qoder and HTTP decision scaffolds constitute different agent configurations, so comparisons must account for their orchestration as well as the underlying model.

Evidence and reports. Initial and final state, events, model requests and tool receipts are stored as content-addressed evidence. Report inspection rechecks the declared bindings without reexecuting the target. Qualified evidence supports a current report; a refused binding leaves the historical judgment unchanged, subject to the availability of its artifacts. Suite comparisons additionally require compatible conditions and qualified paired reports.

Cases and governance. World recipes create bounded variants of permission, transaction and clarification tasks. A failure-derived candidate carries its source record, controls, scope and admission decision. Approved revisions can be used in new runs; withdrawal blocks further use while preserving prior evidence. The lifecycle maintains provenance and eligibility as cases change. Recipe parameters vary coverage within an existing task family.

Scenario Studio and the web view. An operator selects a project, describes business intent and rules, reviews proposed dimensions and resource recommendations, and imports or generates candidate cases. Each edit creates an immutable revision. Bailian execution assigns an independent session to each sample and preserves multi-turn history; the Studio Qoder path supports singleturn cases. Reports expose answers, per-turn and per-dimension results, and evidence qualification. Verified failures can inform a new plan through their source records. The chat grader uses exactanswer comparison after trimming whitespace. Consequently, generated cases require an answer check or operator review; stored rubrics are descriptive rather than an open-ended semantic judge.

The workbench also exposes project and target management, saved sets, controlled worlds, repeat runs, governed suites, diagnostic reports, and historical readback.

## 6 EXPERIMENTS

## 6.1 QUESTIONS AND ASSESSMENT UNITS

We examine three questions under fixed execution and assessment protocols. RQ1: Contract semantics. What do goal-only and full-contract checks establish from the same executions? RQ2: Report qualification. Which evidence and binding faults prevent present report use while behavior stays fixed? RQ3: Operator workflows. How does source eligibility govern case admission, and do reviewed multi-turn cases produce evidence-bound reports? A separate retail study examines objective scope in a public benchmark.

Real tool-world executions. E1 allocates 24 administrations: three families (permission, transaction, clarification), two forms (canonical and bounded recipe), two Bailian configurations and two repetitions. Both configurations use the original decision scaffold, tools and observation interface. Requested identifiers are qwen3.8-flash and deepseek-v4.1-flash; receipts retain returned identities and termination fields. Each episode permits eight requests, 512 output tokens per request and 180 seconds, with automatic retries disabled. Intrinsic step budgets and actual first-action fault exposure remain part of the contract. E2 adds eight matched clarification administrations changing trusted-reply availability. The executions use seed 7; parameter variants and repetitions remain within the declared task families. Full request settings and all interrupted outcomes are retained in the artifact.

Independent checks and comparators. A separately implemented checker reads task specifications, initial and final facts, events, provider receipts and externally bound identities. It evaluates the selected world invariants and task obligations without reading product verdicts. Goal-only and hash-only checks provide feature ablations. Positive, violation, missingness and forged-binding con trols validate checker semantics before interpreting model outcomes. Reports distinguish unresolved behavior, invalid bindings and interrupted execution.

Table 2: Actual current-world outcomes. P/F/U count independently eligible, complete episodes only. Goal-only P/F uses the same episodes; interrupted allocations remain separate. Native and independent full-contract labels agree on all 30 complete episodes.
<table><tr><td>Condition / family</td><td>Allocated</td><td>Complete</td><td>Full P/F/U</td><td>Goal P/F</td></tr><tr><td>E1: permission</td><td>8</td><td>8</td><td>8/0/0</td><td>8/0</td></tr><tr><td>E1: transaction</td><td>8</td><td>8</td><td>8/0/0</td><td>8/0</td></tr><tr><td>E1: clarification</td><td>8</td><td>8</td><td>7/1/0</td><td>7/1</td></tr><tr><td>E2: absent trusted reply</td><td>8</td><td>6</td><td>0/1/5</td><td>0/6</td></tr><tr><td>Total</td><td>32</td><td>30</td><td>23/2/5</td><td>23/7</td></tr></table>

Evidence and binding interventions. E3 uses six predetermined E1 anchors: the first Qwen repetition of each form, regardless of outcome. Thirty host views alter raw availability, byte integrity, current instrument binding, registration binding or snapshot scope; eighteen direct views alter specification, seed or nested run binding. Isolated copies are inspected intact, edited and restored to the original semantic row values and evidence bytes. Current qualification, historical readability and independent eligibility are separate measures; the 48 views are not new agent executions.

Case lifecycle and Scenario Studio. The conditional E4 protocol selects the earliest eligible failure per family, then previews, proposes, validates, activates, consumes and withdraws a supported nonduplicate recipe. Without an eligible source, consumption remains unallocated. Studio instead uses six authored two-turn store-language programs, independently derived preexecution answers and immutable reviewed revisions. Both configurations run them, allocating 24 target answers. Planning, execution, software controls and offline checks retain separate units.

Detailed task declarations, controls, fault edits and lifecycle prerequisites accompany the frozen study manifest. The historical retail setup, its independent three-obligation return reference and its custom training-source split appear in Appendices B–E.

## 6.2 ACTUAL EXECUTION, PERSISTENT EFFECTS AND ASSESSMENT

All 32 allocated tool-world administrations were attempted. Thirty completed; two ended with an idempotency conflict when an agent reused a tool-budget key with different request parameters. All 93 actual provider requests returned and settled. The original errors remain in the allocation, without replacement trials.

The system and independently implemented full-contract checks agree on 23 PASS, two FAIL and five UNKNOWN records among the 30 complete episodes (Table 2). A correction to the independent checker’s handling of refused actions was applied uniformly to all records, without changing any execution (Appendix G.2). Goal-only checking yields the same 23 passes but classifies all remaining complete episodes as failures. The distinction in this collection concerns unresolved contractual context, rather than goal attainment with a demonstrated process violation.

A requested action is not a committed effect. The first permission trace exhibiting authorization recovery has caller alice, but its first resolution request names audit as resolver. The world refuses the request, leaving identical pre/post state digests. A subsequent request names alice, emits the authorized event and resolves the ticket. The original assessment is PASS. Retaining proposals and actual effects explains this outcome: a forbidden request was blocked before it could violate the persistent-state contract. Final-state scoring omits the recovery; counting every proposal as an executed write misdescribes it.

## 6.3 A TRUSTED REPLY CHANGES WHAT CAN BE ESTABLISHED

Of eight reply-availability pairs, six complete and two retain an interrupted absent-reply member. Five complete pairs change from PASS to UNKNOWN; one remains FAIL. In the five unresolved episodes, the agent requests clarification but receives no bound trusted answer. The unconditional documentgoal ablation marks all six absent-reply episodes FAIL. The partial contract instead distinguishes missing context from failure to request clarification. UNKNOWN neither establishes document creation nor overrides binary goal failure. Target stochasticity prevents attributing the contrast solely to reply availability.

![](images/91fc4c8075f09c8c3bc5ed2f903dc25249557b38d2a98d4831f002637b302adf.jpg)

![](images/fc174fb03de9567a1c13fc85752e8ba198999cd35b96998d445bff92e5f50500.jpg)  
Figure 3: Present qualification depends on more than intact bytes. (a) Original host report availability and historical readability on five faults. (b) Hash-only and full-binding eligibility on all eight faults; right-edge labels list their respective counts. Each fault has six fixed anchors; counts are edited evidence views, not agent executions. H1–H5 and S1–S3 are specified in Appendix G.3.

Of the 30 complete records, 25 have definite labels. Equation 5 gives an unresolved-status envelope of $[ 2 3 / 3 0 , 2 8 / 3 0 ] = [ 7 6 . 7 \% , 9 3 . 3 \% ]$ and a decided-label ratio of 23/25. The envelope accounts for five unresolved statuses. Because the absent-reply contract is partially defined, these endpoints do not identify a binary satisfaction rate for the same trajectories. The two interrupted administrations remain outside this conditional denominator.

## 6.4 A RETAINED JUDGMENT AND ITS PRESENT USE CAN DIVERGE

The 48 registered-run interventions comprise 30 host views and 18 direct-verifier views, each observed intact, edited and restored. All intact views are independently eligible; all edited views lose full binding eligibility. The host withdraws current report availability in all 30 edited views. Under the six instrument-binding edits, historical judgments remain readable while current reports are unavailable (Figure 3a). This is the distinction between preserving what was recorded and permitting its present use.

Hash-only checks accept 36 of the 48 edited views: their bytes remain consistent while the specification, seed, registration, instrument or scope no longer matches the assessment (Figure 3b). Full checks instead distinguish six unavailable views from 42 invalid bindings. Restoring each view recovers its original semantic evidence and report outputs. The direct verifier treats edited views as unavailable rather than behavioral failures. Report restoration is also distinct from qualification for cross-agent comparison, which remains unsupported under this study’s registered-runtime interface.

## 6.5 QUALIFIED SOURCES CONSTRAIN CASE MAINTENANCE

Source admission examines the two full-contract failures. Both lack the explicit requester reply required by the clarification recipe, so neither qualifies as a source. The admission rule therefore prevents unsupported case creation. Activation, subsequent use and withdrawal are implemented and covered by software controls, but this allocation provides no eligible source on which to measure their forward effect.

## 6.6 A REVIEWED CASE BECOMES A REAL MULTI-TURN REPORT

Scenario Studio completed an intent-planning call and executed the same reviewed revision with both configurations. All 12 two-turn samples produced reports, and all 24 target answers matched the program answers derived before execution. The actual first answer entered each second-turn history; sessions remained isolated across samples. Four primary dimensions partition the six programs: isolation and missing-value handling each have two cases, while overwrite and readback have one each. Figure 4 illustrates the reviewed-case and conversation interfaces. These results validate execution and report binding for the defined program set, with operator-declared answers checked separately by the research interpreter.

## 6.7 HISTORICAL PUBLIC-BENCHMARK STUDY: WHY OBJECTIVE SCOPE MATTERS

The separately frozen retail study comprises 96 calibration and 192 held-out executions on eight and 16 actor/order components from the τ -bench training split. Each component has three target configurations, two instruction conditions and two repetitions. The declared return subgoal checks order status, item set and refund method; native grading assesses the broader task.

Independent subgoal implementations agree on 187 satisfactions and five violations in all 192 holdout executions, with no unresolved references. Native grading reports 177 successes and 15 failures: ten achieved return subgoals coexist with full-task failure. One retained trace completes a thermostat return while omitting a separate mouse-order modification. This explains a scope distinction without treating either contract as universally preferable.

The instruction contrast retains 96 complete pairs and five discordances. Qwen has 31/32 successes in each condition, although two paired outcomes reverse in opposite directions. Condition-indexed records expose this difference hidden by equal aggregates. Target and simulated-user stochasticity prevent attributing it solely to wording.

With required goal fields withheld in 192 evaluator-facing copies, the raw checker assigns FAIL to every copy, while the completeness-aware adapter assigns UNKNOWN. Original behavior is fixed. A separate sweep withholding 96 views leaves 92 established successes and the finite satisfaction interval [47.9%, 97.9%], expressing unresolved information. Appendix I gives allocation, paired outcomes and masking results.

## 6.8 EXECUTION AND ASSESSMENT COSTS

Tool worlds and Studio use 118 returned requests: 93 world, 24 answer and one planning request. Offline reassessment adds no inference. The retail study records 2,626 target and 1,823 simulated-user requests over approximately 107 minutes at concurrency four (Appendix I). Capturing behavior once permits repeated assessment of its scope and evidence.

## 7 DISCUSSION

An operational unit of assessment. The contract-bound record connects execution, interpretation and reuse. Goal attainment, resolved obligations and present qualification can differ; the workbench exposes these distinctions before an operator compares results or admits a new case.

Scope of the evidence. The tool-world study covers three bounded families, and the retail study assesses a return subgoal on a custom training-source split. Independent checks validate declared specifications under trusted capture assumptions. Studio tests fixed exact-answer programs; case admission yields no eligible source for a forward effect estimate. Broader task generation, open-ended grading and cross-agent comparisons require further contracts and evidence.

## 8 CONCLUSION

AGENTEVAL connects task contracts, execution evidence and report qualification within one evalua tion workflow. Tool-world studies examine committed effects and conditional obligations; evidence interventions expose faults beyond byte integrity. Reviewed multi-turn cases and the retail study connect these mechanisms to operator use, objective scope and missing evidence. The resulting assessment record supports inspection, reuse and maintenance of stateful agent evaluations.

## REPRODUCIBILITY STATEMENT

Task contracts, configurations, execution records, independent checks and analysis inputs accompany the reproduction materials. Appendix G specifies the tool-world protocol and evidence interventions; Appendix H gives the reviewed multi-turn cases. Appendices B–E document the retail study, and Appendix I reports its numerical analyses. Software identities, implementation maps and engineering validation records are provided separately in the reproduction package.

## ETHICS STATEMENT

The studies use simulated business worlds and public benchmark tasks. Tool actions do not operate on real business accounts, and model inference uses authorized providers. Released materials exclude credentials and retain benchmark attribution. Assessments concern the declared contracts and controlled conditions; extrapolation to consequential deployment requires additional evidence.

## STATEMENT ON THE USE OF AI TOOLS

Generative AI tools assisted formalisation and mathematical derivations, experimental design and implementation, result analysis, literature review, manuscript preparation, and illustrative figure creation. Numerical results were checked against retained execution records using deterministic verification. The author is responsible for the final methods, evidence, and conclusions.

## REFERENCES

Anthropic. Demystifying Evals for AI Agents, 2026. URL https://www.anthropic.com/ engineering/demystifying-evals-for-ai-agents. Engineering article, January 9, 2026.

Negar Arabzadeh, Siqing Huo, Nikhil Mehta, Qingyun Wu, Chi Wang, Ahmed Awadallah, Charles L. A. Clarke, and Julia Kiseleva. Assessing and Verifying Task Utility in LLM-Powered Applications. arXiv preprint arXiv:2405.02178, 2024. doi: 10.48550/arXiv.2405.02178. URL https://arxiv.org/abs/2405.02178v2.

Hongliu Cao, Ilias Driouich, and Eoin Thomas. Beyond Task Completion: Revealing Corrupt Success in LLM Agents through Procedure-Aware Evaluation. arXiv preprint arXiv:2603.03116, 2026. doi: 10.48550/arXiv.2603.03116. URL https://arxiv.org/abs/2603.03116.

Edoardo Debenedetti, Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc Fischer, and Florian´ Tramèr. AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents. In Advances in Neural Information Processing Systems, volume 37, pp. 82895– 82920. Curran Associates, Inc., 2024. doi: 10.52202/079017-2636.

Shanshan Gao and Liyi Zhou. Can Agent Benchmarks Support Their Scores? Evidence-Supported Bounds for Interactive-Agent Evaluation. arXiv preprint arXiv:2605.10448, 2026. doi: 10.48550/ arXiv.2605.10448. URL https://arxiv.org/abs/2605.10448.

Aayush Gupta. ReliabilityBench: Evaluating LLM Agent Reliability Under Production-Like Stress Conditions. arXiv preprint arXiv:2601.06112, 2026. doi: 10.48550/arXiv.2601.06112. URL https://arxiv.org/abs/2601.06112.

Isha Gupta, Kai Fronsdal, Abhay Sheshadri, Jonathan Michala, Jacqueline Tay, Rowan Wang, Samuel R. Bowman, and Sara Price. Bloom: An Open Source Tool for Automated Behavioral Evaluations, 2025. URL https://alignment.anthropic.com/2025/ bloom-auto-evals/. Technical report, December 19, 2025.

Michael Hardy, Ruhana Azam, Anka Reuel, Mykel Kochenderfer, and Sanmi Koyejo. Agent Evaluation Reliability: More Tasks Won’t (Always) Fix An Agent Leaderboard. arXiv preprint arXiv:2610.00651, 2026. doi: 10.48550/arXiv.2610.00651. URL https://arxiv.org/abs/ 2610.00651.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? In International Conference on Learning Representations, 2024.

Jannik Kossen, Sebastian Farquhar, Yarin Gal, and Tom Rainforth. Active Testing: Sample-Efficient Model Evaluation. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 5753–5763. PMLR, 2021. URL https://proceedings.mlr.press/v139/kossen21a.html.

Ido Levy, Ben Wiesel, Sami Marreed, Alon Oved, Avi Yaeli, Nir Mashkif, and Segev Shlomov. ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents. arXiv preprint arXiv:2410.06703, 2024. doi: 10.48550/arXiv.2410.06703. URL https:// arxiv.org/abs/2410.06703.

Jiapeng Li. Frozen Judges, Moving Agents: Version-Dependent LLM-Judge Error and the Limits of Judge-Assisted Agent Evaluation. arXiv preprint arXiv:2609.34198, 2026. doi: 10.48550/arXiv. 2609.34198. URL https://arxiv.org/abs/2609.34198v2.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. AgentBench: Evaluating LLMs as Agents. In International Conference on Learning Representations, 2024.

Xing Han Lù, Amirhossein Kazemnejad, Nicholas Meade, Arkil Patel, Dongchan Shin, Alejandra Zambrano, Karolina Stanczak, Peter Shaw, Christopher J. Pal, and Siva Reddy. AgentRewardBench:´ Evaluating Automatic Evaluations of Web Agent Trajectories. arXiv preprint arXiv:2504.08942, 2025. doi: 10.48550/arXiv.2504.08942. URL https://arxiv.org/abs/2504.08942.

Weijun Luo, Kelvin Luu, Xinyi Liu, Guangze Luo, Miguel Romero Calvo, Soham Dan, Daniel Yue Zhang, Ying Liu, and Mohamed Elfeki. Maintaining Benchmarks Against Increasingly Capable Agents: Detection and Remediation of Unearned Passes. arXiv preprint arXiv:2609.34262, 2026. doi: 10.48550/arXiv.2609.34262. URL https://arxiv.org/abs/2609.34262.

Promptfoo. Promptfoo: Test Case Configuration. Software documentation, n.d. URL https:// www.promptfoo.dev/docs/configuration/test-cases/. Accessed October 12, 2026.

Stephan Rabanser, Sayash Kapoor, Peter Kirgis, Kangheng Liu, Saiteja Utpala, and Arvind Narayanan. Towards a Science of AI Agent Reliability. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pp. 102769–102820. PMLR, 2026. URL https://proceedings.mlr.press/v306/rabanser26a.html.

Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pp. 4902–4912. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.acl-main.442. URL https://aclanthology. org/2020.acl-main.442/.

Mohammadreza Sediqin, Shivali Dalmia, Srinivasa Karthikeya Reddy Kovvuri, and Abhishek Mukherji. A Trust Layer for Agent Evaluation. arXiv preprint arXiv:2610.07274, 2026. doi: 10.48550/arXiv.2610.07274. URL https://arxiv.org/abs/2610.07274.

Hyewon Suh, Binfei Ji, Seojune Lee, Rishi Khare, Basit Khan, Hyunjun Kim, Tianyi Zhang, Venkat Krishna Srinivasan, Peter Belcak, Shizhe Diao, Pavlo Molchanov, Yingyan Celine Lin, and Zhen Dong. AgentSuite: Toward More Reliable Agent Evaluation with a Component-Based Benchmark Auditing Pipeline. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pp. 116300–116319. PMLR, 2026. URL https://proceedings.mlr.press/v306/suh26a.html.

UK AI Security Institute. Inspect: Eval Sets. Software documentation, n.d. URL https:// inspect.aisi.org.uk/eval-sets.html. Accessed October 12, 2026.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv preprint arXiv:2406.12045, 2024. doi: 10.48550/arXiv.2406.12045. URL https://arxiv.org/abs/2406.12045.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36, pp. 46595–46623. Curran Associates, Inc., 2023. doi: 10.52202/ 075280-2020.

YiShan Zheng, Yuan Wu, and Yi Chang. ToolRobustBench: Stage-Wise Perturbation Evaluation and Failure Diagnosis for Tool-Calling Agents. arXiv preprint arXiv:2608.23635, 2026. doi: 10.48550/arXiv.2608.23635. URL https://arxiv.org/abs/2608.23635.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. WebArena: A Realistic Web Environment for Building Autonomous Agents. In International Conference on Learning Representations, 2024.

## A NOTATION AND FINITE EVIDENCE BOUNDS

Objects and scope. A identifies the target configuration; x the task; c the condition; ξ the full episode; and e a retained evidence view. The declared contract Γ contains a nonempty obligation set J; a goal-only subset can omit process requirements when context fully defines the same goal. Raw software records preserve canonical status and reasons. An objective’s name, version, and required evidence are part of the assessment.

Derivation of Equation 5. For a fixed collection of N admissible records, assigning the n<sub>?</sub> unresolved statuses binary values gives the accounting envelope in Equation 5. To interpret it as a satisfaction identification bound, every record must have a defined latent binary outcome and every established label must be sound. Under those additional assumptions, let U contain the unresolved outcomes. The finite satisfaction rate is

$$
\frac { n _ { + } + \sum _ { i \in U } Y _ { \Gamma , i } ^ { * } } { N } .
$$

The unresolved sum lies between zero and $n _ { ? }$ . If each unresolved binary value can vary independently over {0, 1} among admissible completions, both endpoints are attainable; otherwise the envelope remains an outer bound. No missing-at-random assumption is needed. Historical retail masking satisfies the defined-binary-outcome premise: it withholds observations of fixed return predicates. The current absent-reply clarification contract is partially defined instead, so its envelope reports unresolved status mass without claiming either endpoint as a possible satisfaction rate of the same fixed trajectory. Qualification refusals and interrupted administrations retain separate categories.

Dependence and extrapolation. Repeated executions, wording variants, and models evaluated on one actor/order component do not create independent tasks. Components in this study also share a world and catalog. Resampling components describes sensitivity within this design; it is not a proof of population coverage. For an untested condition, two possible systems can agree on all observed runs and disagree arbitrarily elsewhere. Further capability claims require new evidence or explicit assumptions linking those conditions.

Causal interpretation. A difference across observed conditions is a response contrast. Attribution to an internal mechanism additionally requires defined interventions, suitable controls, and identifying assumptions. A recorded intervention or sealed receipt documents an operation; it does not establish those assumptions.

## B HISTORICAL RETAIL CONTRACTS AND EXECUTION PROCEDURE

Declared retail subgoal. The assessed return subgoal comprises the requested order status, returned item set, and refund-method identifier in native simulated state. It does not establish financial settlement, complete dialogue semantics, or every action requested in the surrounding instruction. The wording condition changes the return clause while retaining its slots and prefix. Some preserved prefixes contain another order request; full-task native reward and return-subgoal satisfaction consequently need not agree.

Scope annotation. The original executions retain their declared return subgoal. The post-freeze annotation correction is described in Appendix D.

Canonical status and evidence gate. The historical adapter’s canonical checker exposes PASS, FAIL, PARTIAL, UNKNOWN, and INVALID. Its composed verifier can return UNKNOWN when a child is unresolved, even if another child fails. A supplied state with a missing required path can yield FAIL, whereas an unavailable state yields UNKNOWN. These are raw component semantics.

The evaluated adapter additionally checks capture completeness and researcher-withheld required fields. For an unusable view, it supplies no state to the checker; raw and gated results are both retained. Analysis maps PASS to established conjunction satisfaction, FAIL/PARTIAL to non-satisfaction, and UNKNOWN/INVALID to unresolved, preserving all original statuses. This mapping is specific to the declared conjunction and does not reinterpret partial credit as full success.

## Procedure.

1. Freeze task components, subgoal specifications, target configurations, conditions, budgets, and analysis rules. Keep reference data outside target inputs.

2. Reset the native environment and execute within the declared request, step, retry, and wall-clock limits.

3. Record actual tool events and state before native grading; retain native outcomes independently of the scoped verifier.

4. Evaluate identical evidence with the canonical checker and separately implemented reference. Apply the declared measurement gate and retain its inputs and reasons.

5. Perform evidence masking on copies only. Report original and manipulated views separately; never count a missing measurement as an observed behavioral violation.

6. Aggregate within the declared units and conditions, retain unknowns, and attach objective, evidence, and analysis versions to the report.

## C RETAIL ADAPTER AND EVIDENCE CAPTURE

The retail study uses an adapter to the native benchmark environment. Its execution and assessment protocol is distinct from the tool-world study. The reproduction manifest identifies the adapter, benchmark source and analysis environment.

Checks and evidence. The specification checker dispatches predicates, records check-level reasons, selects evidence visibility and aggregates statuses. A separate oracle-composition interface has its own failure and abstention precedence. The study uses the specification checker with the capturecompleteness gate in Appendix B. Execution is recorded before native grading so that the benchmark comparator and return-subgoal reference can inspect the same agent effects.

Native bridge and capture. The adapter captures initial and final native state, tool actions and state deltas before replaying the benchmark’s reference actions. Native full-task grading and the returnsubgoal reference then assess their respective contracts on this execution record. The tool-world host provides the separate execution and report interfaces described in Appendix G.

## D HISTORICAL REFERENCE CONSTRUCTION AND ASSESSMENT SCOPE

Independent implementation. The reference checker evaluates frozen instruction-derived return clauses on captured state using a separate implementation. It does not read AES verdicts or native rewards to establish its label. The clauses identify requested return status, the returned item set, and the specified refund method. Source instructions, initial state, contract versions, and evidence locations accompany each judgment. Agreement between implementations tests the shared objective’s execution; it does not establish that the objective includes every legitimate task obligation.

Native outcome. The native comparator evaluates its own expected full-database result. Its reference actions and output are kept outside target execution, and actual state is captured before reference replay. The raw native outcome remains separate from return-subgoal satisfaction. A discrepancy may reflect an extra requested action, an obligation omitted from our subset, a comparator defect, or a task interpretation issue. It is not assigned to evaluator error without examining its cause.

Post-freeze scope correction. A review prompted by a holdout discrepancy led to a census of all frozen instructions. Three include additional order requests in the preserved prefix. The scope annotation was corrected to return-bearing requests assessed on a declared return subgoal. The original tasks, splits, specifications, and assessments were retained. An unused ancillary non-targetorder preservation annotation was invalidated for those cases. This correction is descriptive and post-freeze, not a new preregistered result or an outcome-based exclusion.

Unresolved obligations. Informed confirmation semantics, unsupported verbal claims, disclosure, and external settlement are not established by these state clauses. Broader policy gold remains unknown.

Review provenance. The reference is constructed from source instructions and captured state. Agreement measures consistency between software implementations, not agreement between human annotators. Implementations can share conceptual errors; objective scope and disagreement records therefore accompany the numerical summaries.

## E HISTORICAL ALLOCATION, DEPENDENCE, AND ANALYSIS

Units and splits. The frozen allocation contains 24 mutable actor/order components: eight calibration and 16 custom holdout components. Three models, two wording conditions, and two repeats produce 96 and 192 opportunities, respectively. Components retain all owned orders for dependency checks, including orders named by additional requests. They share a catalog and world; graph separation does not make them IID draws from deployment tasks. The custom holdout is drawn from the upstream training split and is not an official benchmark test partition.

Denominators. Allocated, completed, reference-decidable, and unresolved counts are retained by split, model, and condition. Terminal infrastructure failures and missing or malformed evidence remain unresolved; unfinished jobs are not imputed as observations. Retries stay in the attempt ledger for accounting and do not create new independent tasks. Success counts over all allocated opportunities are not interchangeable with decided-only success or conditional decision coverage.

Status projection. Analysis preserves raw statuses and maps PASS to conjunction success, FAIL/PARTIAL to conjunction failure, and UNKNOWN/INVALID to unresolved. The same projection is applied to the raw and completeness-gated outputs. The independent reference retains its own three-valued decisions. Native B0 is a separate estimand; matched B4 shares the canonical verifier, specification, evidence, and gate.

Archived masked assessments also contain a scalar score placeholder of zero when their status is UNKNOWN. This field is not an observed behavioral score. The reported missingness counts and bounds derive from the typed status and independent reference, preserving unresolved outcomes rather than averaging those placeholders as failures.

Reference-decidable comparisons. Agreement is reported only on records where the independently implemented objective reference is decidable, with the unresolved reference count retained separately. Equality with the matched implementation control tests consistency under the same specification; comparisons against native grading retain the different assessment scope.

Paired wording analysis. C0 and C1 are paired within component, target model, and repetition. The analysis retains success-to-failure and failure-to-success discordances, missing pairs, and withincomponent mean differences. Any exploratory percentile interval uses 2,000 component resamples with seed 20260920, preserving paired observations; no interval is reported with fewer than two components. The interval describes sensitivity conditional on the chosen task family rather than population coverage.

Evidence masking. The core copies withhold the frozen required subgoal fields; no new trajectory is executed. The separate final holdout sweep, specified before final reference-label analysis, uses rounded counts 0, 19, 58, and 96 of 192 for its nominal fractions. It compares whole-state and first-status-field removal under random and outcome-linked stress mechanisms. Its timing and protocol version are distinct from the initially frozen all-required-fields copy test. Actual counts and denominators accompany every plot. Standard finite bounds quantify unresolved mass under sound-label assumptions; they are not confidence intervals. No model-selection experiment, Type-II causal test, or broad missingness-mechanism comparison is inferred from this analysis.

Table 3: Study contracts and observation units. Each experiment retains the protocol and evidence needed to interpret its results
<table><tr><td>Study</td><td>Assessment focus</td><td>Contract and unit</td></tr><tr><td>Tool worlds</td><td>Stateful execution</td><td>Permission, transaction and clarification; complete or interrupted episode</td></tr><tr><td>Evidence interventions</td><td>Report qualification</td><td>Binding and availability; edited evidence view</td></tr><tr><td>Retail benchmark</td><td>Objective scope</td><td>Three return-state obligations; component/configuration/repeat</td></tr><tr><td>Checker controls</td><td>Instrument validation</td><td>Declared positive, violation, missingness and binding controls</td></tr></table>

## F ARTIFACTS AND REPRODUCIBILITY

Versioned inputs. The artifact records canonical system and native benchmark commits, dependency environments, task/component identities, splits, target and simulator models, conditions, budgets, contracts, and analysis versions. Native training-split provenance and the custom study split are both retained. The historical retail ledger retains configured service aliases and execution times; it does not retain request-level response model identifiers or finish reasons. The current study uses original provider receipts to retain returned model identifiers, response IDs and finish reasons. Neither an alias nor a matching returned identifier establishes immutable provider weights.

Execution evidence. Each run links its frozen job to initial state, actual pre-grader final state, tool actions, observations, per-step changes, native outcomes, canonical assessments, independent reference checks, and usage. Digests support consistency checks between these artifacts. State-diff reconstruction checks internal continuity; it is not a claim that the agent was independently reexecuted. Interrupted and failed attempts remain inspectable rather than being replaced by successful reruns.

Resource units. Usage separates target and user-simulator requests and records provider-reported tokens, attempt identity and latency. The historical records omit request-level finish reasons. Among simulator requests, 116 completion-token counts equal the 2,048-token cap; token counts alone cannot establish whether a response was truncated. Cache-hit and reasoning-token breakdowns are also unavailable. Evidence analysis adds no model requests, and the study uses deterministic references rather than a model-judge or human-labeling stage.

Historical tokens and execution windows are recorded resource units rather than an invoiced currency bill. Shared-trace analysis adds no target calls, while acquisition and assessment retain distinct costs.

Reproduction and release. A study manifest links the execution records, task and analysis defini tions, and source identities needed for reproduction. Tables and numerical figures are generated from the corresponding result manifests. Current tool-world observations and historical retail observations remain separate throughout analysis. Comparisons with other studies require compatible objectives and execution conditions.

The reproduction package includes implementation maps, engineering controls and screenshot provenance alongside the scientific records. These materials support inspection of the system and research instruments; their software-test counts are not agent-performance observations.

## G TOOL-WORLD ARCHITECTURE AND ASSESSMENT CONTRACTS

The execution host connects target registration, model decisions, world actions, evidence capture and report inspection. Table 3 distinguishes the contracts and observation units of the studies; Table 4 maps the operator workflow to its outputs. Complete software identities and the module-to-source mapping are retained in the reproduction package.

Execution host and providers. The host registers a target configuration and supplies model decisions to the world through the declared action interface. Provider adapters project allowed observations and bind decisions to their response records. Bailian uses an HTTP decision scaffold; Qoder uses a constrained native MCP session. The host enforces scope, budgets and deadlines and inspects existing records during recovery to avoid silently repeating an administration.

Table 4: Implemented workflow and evidence. Each stage exposes a concrete artifact or decision. Tool-world execution and exact-answer chat assessment use different task contracts.
<table><tr><td>User operation</td><td>Product component</td><td>Retained output</td></tr><tr><td>Define intent and cases</td><td>Scenario Studio; imports; re- views</td><td>Plan, candidate provenance, immutable reviewed revision</td></tr><tr><td>Configure and execute</td><td>Application SDK; registered host; local worlds</td><td>Frozen target/task binding, budget and actual requests</td></tr><tr><td>Act in a stateful world</td><td>Permission, transaction, clari- fication kernels</td><td>Visible tool receipts; privileged state and event evidence</td></tr><tr><td>Inspect an assessment</td><td>Verification registry; CAS; re- port host</td><td>Check-level results, qualification diagnosis, historical record</td></tr><tr><td>Compare and maintain</td><td>World suites; governed case lifecycle</td><td>Compatible conditions, approval, consumption and withdrawal receipts</td></tr></table>

World and report contracts. Task definitions declare permission, transaction and clarification obligations. Recipe compilation supplies bounded variants and family-specific controls. World checks evaluate state and event predicates under an identified measurement dependency closure. Report inspection reopens captured artifacts without a target execution interface, separating current qualification from the recorded judgment. Historical readback requires its supporting artifacts to remain available.

Governed assets and user-facing cases. Case maintenance links an eligible failure to proposal, calibration, activation, subsequent use and withdrawal. Admission checks include semantic duplicate detection. Suite comparison additionally requires compatible world conditions. Scenario Studio supports planning, candidate import or generation, reviewed revisions, answer-harness execution and failure-informed planning; its exact-answer contract is distinct from world-state predicates.

Other system modules. Typed check registries, snapshots, receipts and content-addressed storage provide shared infrastructure. Target access and observability constrain the supported interpretation of reports. Probe construction, evaluation selection, memory and governance modules support broader evaluation workflows; the experiments address the specific execution and assessment paths described here.

## G.1 DECLARED WORLD AND REFERENCE CONTRACTS

The complete contract is the selected original profile together with the family task. Transaction tasks select nine profile members: calendar, commerce, document, identity, mail and ticket invariants; refund conservation; cached-read consistency; and write attribution. Permission and clarification select seven: document, identity and ticket invariants; refund conservation; read-only actor authority; cached-read consistency; and write attribution. These names denote concrete predicates rather than unrestricted financial or causal certification.

Conservation compares recorded refunds with privileged settled value. Authority is computed from the original identity/delegation rules. Cache consistency compares existing live records with retained reads. Attribution checks changed state leaves against the full serialized receipt ledger and its recorded path prefixes; uncovered changes with other recorded causes can remain unresolved. The independent implementation preserves those exact selected semantics, including empty-domain, missing-fact and numeric-type boundaries.

The checker consumes exact CAS bytes, the original execution graph and an external declaration. Constructor-derived task, specification, prepared state, tools, observation rules and run conditions are sealed before target access. Source-normalized registration metadata is checked against its complete predeclared blueprint. Opaque attempt identifiers are observed and cross-bound rather than represented as preknown. Eligibility, terminal completion, behavioral decision and historical readability remain separate fields; poisoning stored result labels cannot supply independent truth.

Table 5: Family obligations beyond apparent goal state. Canonical and recipe forms retain their distinct frozen specifications.
<table><tr><td>Family</td><td>Additional declared task evidence</td></tr><tr><td>Permission</td><td>Actual caller resolves the target; identity authority and audit remain intact. Recipe tasks also preserve the protected ticket and reject a resolver-identity claim inconsistent</td></tr><tr><td>Transaction</td><td>with the caller. Exactly the declared refund matches independently settled value, preserves unrelated order/refund state and records the required refund rollback exposure. The first-action</td></tr><tr><td>Clarification</td><td>fault is retained even if spent on a different allowed action. A request and trusted owner reply are bound to the task and token before the one permitted document creation. Recipe tasks also preserve the protected document. Missing trusted context follows the original abstention rule.</td></tr></table>

The registered local interface supports assessment of captured executions under the declared research contract. It does not attest an external agent implementation or authorize an unrestricted crossagent comparison. Provider responses identify returned service configurations without establishing immutable provider weights.

Independent checker validation. The independent checker is tested with positive controls, known violations, missing evidence and forged bindings. Controls cover canonical and recipe permission rules, transaction replay, clarification-token binding, absent replies, and known violations accompanied by unrelated missing facts. Registration checks use predeclared metadata independently of stored verdicts.

Composite and execution-path controls compare the selected world predicates, events, observations and budget histories against their declared semantics. These software controls validate the research instrument separately from real-model episodes. Implementation details and control inventories accompany the reproduction materials.

The checker is then applied uniformly to the captured study records. Controlled edits of registered-run evidence test present report qualification in Appendix G.3; they are distinct from the software controls used to validate checker implementation.

## G.2 EXECUTION PROTOCOL AND INDEPENDENT ASSESSMENT

The study fixes allocation order, tasks, configurations, prompts, budgets and registration declarations before execution. Captured requests and responses identify the actual inference operations. Initial and final state, events and content-addressed artifacts support assessment of each administration. Interrupted administrations remain in the allocation and are not replaced.

Evidence readback and analysis operate on preserved execution records without repeating target inference. Recovery continued only the remaining allocated administrations. The reproduction materials retain the original incomplete records as well as the complete episodes.

The trusted boundary consists of the execution host, world state and event capture, provider response records, and the separately implemented checker. Offline inspection uses the original world interfaces without a target entry point. These controls delimit the observed execution protocol; they do not certify operating-system isolation or provider internals.

Post-observation checker correction. The initial independent checker required a returned tool receipt for every selected action. The runner instead records declared-limit or classified-exception refusals as decisions without returned tool calls, and includes the refusal in the next observation. Correcting this mismatch changed six eligibility decisions, five on system-PASS records. The amended checker reconstructs the refused-action execution paths and budget history while preserving state, event and observation consistency. The correction was applied uniformly to all 32 records without new model trials or outcome-based exclusions. Both checker definitions and the changed assessments are retained in the reproduction materials.

Table 6: Current world allocation by requested configuration. Counts preserve all allocated administrations; P/F/U refer only to complete, independently eligible episodes. Provider requests include interrupted administrations.
<table><tr><td>Requested configuration</td><td>Allocated</td><td>Complete</td><td>P/F/U</td><td>Requests</td></tr><tr><td>qwen3.8-flash</td><td>16</td><td>15</td><td>12/0/3</td><td>52</td></tr><tr><td>deepseek-v4.1-flash</td><td>16</td><td>15</td><td>11/2/2</td><td>41</td></tr></table>

Table 7: Actual edited-view eligibility, six views per fault. Native host refusals and unavailable direct verifications are distinct interface results. No direct edited verification returns a behavioral FAIL.
<table><tr><td>Fault</td><td>Edit</td><td>Full binding</td><td>Hash only</td></tr><tr><td>H1</td><td>Remove required raw object</td><td>UNAVAILABLE</td><td>UNAVAILABLE</td></tr><tr><td>H2</td><td>Wrong bytes at the original lookup key</td><td>INVALID</td><td>INVALID</td></tr><tr><td>H3</td><td>Rehashed current instrument/model binding</td><td>INVALID</td><td>ELIGIBLE</td></tr><tr><td>H4</td><td>Registration binding mismatch</td><td>INVALID</td><td>ELIGIBLE</td></tr><tr><td>H5</td><td>Snapshot scope mismatch</td><td>INVALID</td><td>ELIGIBLE</td></tr><tr><td>S1</td><td>Rehashed specification digest</td><td>INVALID</td><td>ELIGIBLE</td></tr><tr><td>S2</td><td>Rehashed seed binding</td><td>INVALID</td><td>ELIGIBLE</td></tr><tr><td>S3</td><td>Rehashed nested raw run scope</td><td>INVALID</td><td>ELIGIBLE</td></tr></table>

Execution-path, predicate and adversarial controls were repeated for the amended checker. Returned dispatches still require their response records; unclassified crashes, inconsistent effects and missing history remain invalid or unresolved. Independent offline reproduction yielded the same assessments, and the two interrupted episodes remained incomplete.

The two interrupted episodes ended with idempotency conflicts and lack complete outcome evidence. Closing an execution attempt does not establish successful task completion. Their partial records are preserved separately from the 30 complete episodes.

Each configuration has one interrupted absent-reply episode. Sixteen initial paired request envelopes match except for the requested model. All 93 provider requests returned; neither interrupted episode arose from a missing network response. Different response paths, request counts and partial-contract labels are descriptive outcomes under the common scaffold, rather than an intrinsic model ranking.

## G.3 ACTUAL EVIDENCE INTERVENTIONS AND RESTORATION

The six anchors are the first Qwen repetition of each E1 family/form, at allocation positions 1, 4, 5, 8, 9 and 16. Selection precedes outcome inspection; all six happen to be PASS. Each anchor receives eight separate edits to isolated copies of the evidence store and registration state. The original execution, responses and source remain fixed. Intact, edited and restored inspection produces 144 primary observations, with 90 additional historical-readback observations for the 30 host views.

H1–H5 use the original host inspection, reporting, metric and history APIs. S1–S3 use the original direct verifier because these nested bindings are not host-level edits. Native refusal is operational unavailability, separate from a behavioral verdict. The goal-only fault ablation gives 39 PASS and nine UNKNOWN; it cannot establish present admissibility. The hash-only ablation checks available object bytes, excluding declared semantic bindings. Historical readability survives H3 in six views; the other 24 host edits make history unavailable. None of these counts are separate agent trajectories.

Restoration recovers the original semantic registration fields and evidence bytes. All 48 restored reference decisions match their intact decisions, and all 30 host grade hashes match. Internal storage revisions are tracked independently of semantic restoration. The generic registered comparison remains unsupported in intact and restored views; restoring an inspectable report does not confer cross-agent comparison qualification.

Offline assessment and numerical reproduction make no model requests. A separate reproduction from the preserved phase inputs reconstructs full-binding, goal-only and hash-only results and compares all 144 primary and 90 historical outputs. Missing objects and mismatched bytes remain part of the reproduced intervention inputs.

## H REVIEWED MULTI-TURN WORKFLOW AND ACTUAL INTERFACE

Scenario Studio received one real intent-planning response and an operator-authored dataset with six fixed integer-store programs. An immutable reviewed revision was executed by both target configurations. Each sample began with an empty store; its actual first answer entered the retained assistant history for the second request. No expected second answer was inserted into the target history. The research interpreter derived each answer from the program before execution. The system retains operator-declared answers; the interpreter independently checks their correctness for this program set.

Table 8: The six Studio programs fixed before execution. A semicolon denotes a newline in the actual request or answer. Each sample has one primary dimension and is executed once per configuration.
<table><tr><td>Sample</td><td>Dimension</td><td>First request → answer</td><td>Second request → answer</td></tr><tr><td>S1</td><td>isolation</td><td>SET shared 7 → OK</td><td>GET shared → 7</td></tr><tr><td>S2</td><td>isolation</td><td>GET shared → MISSING</td><td>SET shared 4 → OK</td></tr><tr><td>S3</td><td>overwrite</td><td>SET x 3; SET x 9 → OK; OK</td><td>GET x → 9</td></tr><tr><td>S4</td><td>missing</td><td>SET y -2; GET y → OK; -2</td><td>GET unset → MISSING</td></tr><tr><td>S5</td><td>readback</td><td>SET a 1; SET b 2 → OK; OK</td><td>GET a; GET b → 1; 2</td></tr><tr><td>S6</td><td>missing</td><td>GET z → MISSING</td><td>SET z O; GET z → OK; 0</td></tr></table>

The scope is exact execution and report binding on these programs. A program or dimension label is not an independent judge; the four dimensions partition the six samples. Planning is a separate resource request and is not counted as an agent answer or as evidence of generated-case quality.

## I HISTORICAL RETAIL RESULTS AND EVIDENCE INTERVENTIONS

## I.1 OBJECTIVE-ALIGNED ASSESSMENT

All 192 held-out executions yield a decidable independent return-subgoal reference, with 0 unresolved references. The declared objective succeeds in 187 runs and fails in five; native full-task grading reports 177 successes and 15 failures. Ten runs therefore satisfy the return objective while failing the broader native assessment. Table 9 resolves these outcomes by target and condition.

Table 9: Holdout return-subgoal outcomes. Each cell reports established successes / allocated opportunities, with unresolved outcomes retained in the denominator. The assessed objective is the declared return subgoal.
<table><tr><td>Target</td><td>CO</td><td>C1 wording</td></tr><tr><td>DeepSeek Flash</td><td>31/32</td><td>32/32</td></tr><tr><td>Qwen Flash</td><td>31/32</td><td>31/32</td></tr><tr><td>Kimi K3</td><td>32/32</td><td>30/32</td></tr></table>

AES and the independent same-subgoal reference disagree on 0 reference-decidable cases. Matched B4 differs from the gated canonical assessment on 0 cases. This agreement establishes consistent objective execution across the integration. Retaining native outcomes alongside it makes the ten cross-scope differences visible and preserves the objective needed to interpret each judgment.

(a) Review the case

![](images/832be0cc5ec0fdb669258a1de5f196a0b47ac786bfb9646e611b2e0ed829bd0b.jpg)  
Figure 4: From a reviewed case to a retained conversation. Cropped views of the actual workbench show (a) the reviewed input and expected answers and (b) the Qwen execution, SET shared 7 → OK followed by GET shared → 7. Interface labels are Chinese; panel headings identify the two workflow stages. Cropping removes browser chrome and unrelated navigation without modifying case content or model answers. Numerical validation uses the captured request and evidence records.

![](images/a171fc3af491e4a2601e8b78d0965d5f8dcc12ae49b13b9642430a1dbb734cdf.jpg)  
Unknown objective references: 0/192. Matched B4 / AES diferences: 0/192.  
192 runs / 16 components. Native and subgoal outcomes answer diferent questions.

Figure 5: Objective scope and wording responses. Native full-task outcomes and declared returnsubgoal outcomes are distinct assessments. Paired C0/C1 outcomes preserve component, model, and repeat; incomplete pairs are shown separately. The panels connect objective-level outcomes with paired condition responses.

## I.2 CONDITION-INDEXED DIAGNOSIS

The wording analysis includes 96 complete pairs, with 5 differing objective outcomes. Qwen records 31/32 successes in each condition, yet two paired outcomes reverse in opposite directions. Equal aggregate scores thus conceal changes that condition-indexed reporting exposes. The paired design matches component, model, and repetition; both target and simulator contribute stochastic variation.

A posthoc case, train:8, illustrates objective-level diagnosis. The run completed the thermostat return but omitted an additional mouse-order modification: native grading failed while both returnsubgoal checks passed. Local replay of native reference actions reproduced the stored comparator state. The preserved trace locates this particular disagreement in an additional request outside the return objective, retaining the request’s ambiguity for inspection.

## I.3 EVIDENCE INTERVENTIONS

The measurement test deletes required goal fields from an evaluator-facing copy while retaining the original state and trajectory. This changes what the evaluator can establish, not what the agent did. The core frozen adapter records both raw canonical and completeness-gated assessments for the same manipulated view.

Among 192 evaluable holdout copies, raw required-field masking produces 192 FAIL/PARTIAL results and 0 UNKNOWN/INVALID results. The completeness-gated path reports 192 unresolved assessments. The reference check likewise uses the declared evidence view; its behavior is not inferred from AES output.

![](images/7d280c212bbda630caf5722b9d4b284df22090ba005ff1a287b0bc05d1a8a917.jpg)  
(a) All required fields per copy. (b) Separate random first-status-field sweep.  
Finite identification bounds, not confidence intervals. No new agent executions.

Figure 6: Raw component versus measurement-aware reporting. Required-field deletion and whole-view absence are different inputs to the canonical verifier. The adapter marks withheld required evidence as unavailable. The separate holdout sweep masks 0, 19, 58, 96 of 192 trajectories, displaying finite identification bounds rather than population confidence intervals.

This comparison exposes an interface requirement. The raw checker treats an absent required path in a supplied state as a failed predicate; it cannot infer that a researcher withheld the field. The adapter carries that information and gates the input. Explicit completeness metadata therefore separates a failed task predicate from an unavailable measurement. Preserving raw statuses and gate reasons makes that distinction visible throughout the reporting path.

A separate exploratory plan, fixed before final reference-label analysis, evaluates 24 configurations: three masking mechanisms, four proportions, and two evidence views. It masks 0, 19, 58, 96 of the 192 trajectories (0.0%, 9.9%, 30.2%, 50.0%). At 50% random first-status-field masking, 92 established successes coexist with a finite interval of [47.9%, 97.9%]. A high observed success count can therefore accompany substantial unresolved information. Failure-linked and success-linked policies stress selective missingness; all views reuse original behavior and add no agent executions.

## I.4 RESOURCE ACCOUNTING

The execution ledger records target and simulated-user requests separately, including tokens, request latency, and recorded attempts. Request-level finish reasons are not retained in the core ledger; 116 simulator completions reach the configured 2,048-token cap, so truncation cannot be resolved from token counts alone. Shared-trace verification and evidence masking require no new model requests; the final masking analysis makes 1,152 local canonical-verifier calls over the recorded evidence. The ledger separates these local assessments from target and simulated-user inference, making the resource requirements of execution and subsequent analysis inspectable.

Table 10: Core study inference usage across calibration and holdout, including recorded retries. Tokens are provider-reported counts; development qualification and the separate AgentDojo case are excluded.
<table><tr><td>Role</td><td>Requests</td><td>Input tokens</td><td>Output tokens</td></tr><tr><td>Target</td><td>2626</td><td>14043657</td><td>428996</td></tr><tr><td>User simulator</td><td>1823</td><td>878831</td><td>736159</td></tr></table>

The core execution windows total 6409.12 s, with 0 recorded task retries. This elapsed total is not the sum of overlapping request latencies. No model-judge or human annotation stage is included.

This separation is operationally useful: objective checks and evidence-view experiments can be repeated after a campaign without rerunning the target. The measured execution windows total about 107 minutes at the frozen concurrency of four. Combined with the preserved request and evidence ledgers, this yields a practical workflow for collecting interactive behavior once and examining its assessment under multiple evidence views.