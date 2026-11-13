# Evidence-Governed Adoption of Self-Generated Updates in Autonomous AI Agents

Subodh Lonkar Independent Researcher

## Abstract

Autonomous agents increasingly modify prompts, memories, tool configurations, workflows, and control policies in response to interaction histories. The ability to generate a plausible modification, however, does not establish that it should become persistent system state. Candidate selection is adaptive, evaluators may be imperfect or manipulable, and repeated local improvements can conceal degradation relative to an earlier operating baseline. This paper develops Evidence-Governed Evolution (EGE), a statistical and operational framework for governing persistent adoption of self-generated agent modifications. EGE separates candidate generation from confirmatory adoption, evaluates frozen candidates on fresh paired audit samples, requires practically meaningful improvement on a prespecified primary objective, protects declared capabilities through reference-relative non-inferiority tests, and applies mandatory governance constraints before bounded canary deployment. The statistical contribution is an evaluator-calibrated simultaneous lower-bound construction for a countable sequence of adaptively generated candidates. It combines candidate-wise Hoefding bounds with independent, candidate-specific calibration samples that compare the operationa evaluator with a prespecified reference evaluation procedure. A second result formalises reference replacement through explicit bridge tests and derives a cumulative degradation bound along a versioned reference lineage. We also give a conservative audit-planning condition that links efect-size margin, evaluation sample size, calibration sample size, and a lower bound on acceptance probability. A reproducible Monte Carlo stress test illustrates the importance of calibration under candidate-dependent evaluator optimism; it is explicitly a stylised statistical experiment, not an evaluation of a deployed agent. The paper does not claim empirical validation on a real autonomous-agent system. All guarantees are conditional on bounded paired scores, fresh sampling, a valid reference-evaluation construct, and the declared evaluation distributions; they do not establish universal agent safety or robustness to arbitrary distribution shift.

Keywords: autonomous agents; self-evolving agents; adaptive data analysis; sequential inference; evaluator calibration; non-inferiority; AI governance.

## 1 Introduction

Autonomous agents are systems whose behaviour depends not only on foundation-model weights but also on prompts, memory, tools, workflows, planning procedures, routing policies, and execution constraints. An agent can therefore change its behaviour without modifying its underlying model. Candidate changes may be generated from execution traces, user feedback, self-reflection, search, or optimisation.

A candidate that performs well on recently encountered tasks is not necessarily a reliable improvement. It may overfit to development data, reduce performance on previously supported tasks, increase latency or resource use, or violate operational constraints. The risk is amplified when an agent repeatedly proposes and deploys changes with limited human intervention.

Prior work studies reflection, iterative refinement, and skill acquisition in language-model agents (Shinn et al., 2023; Madaan et al., 2023; Wang et al., 2023). Automated prompt optimisation and evolutionary search provide additional candidate-generation mechanisms (Guo et al., 2024). These methods motivate the ability to generate changes; they do not by themselves settle whether a proposed change should become persistent system state. Broader work on AI safety identifies specification failures, distributional failures, and unsafe exploration as recurring challenges (Amodei et al., 2016; García and Fernández, 2015).

The adoption problem has several properties that make it diferent from ordinary ofline model selection. First, candidate proposals are often adaptive: the next proposal may depend on previous successes, failures, evaluator feedback, or deployment traces. Second, the candidate can modify the mechanism that will later produce evidence, for example by changing a prompt, retrieval policy, tool-selection rule, or evaluator-facing output format. Third, the system may optimise a narrow objective while being expected to preserve a much broader capability profile. Fourth, deployment is not a purely statistical event. A candidate can pass a benchmark and still fail under production load, interact unexpectedly with tools, or create external side efects that cannot be undone by restoring an earlier configuration.

These properties motivate a layered adoption design. The proposal mechanism is permitted to explore the design space, but it does not control the confirmatory audit process. The audit process measures a frozen candidate against prespecified comparators and margins. A governance layer enforces requirements that must not be traded against average task performance. Finally, an operational deployment layer limits exposure while collecting evidence about behaviour in the target environment. The layers are related but not interchangeable: a confidence bound cannot replace a permission boundary, and a canary cannot repair an invalid statistical comparison.

The scope of EGE is deliberately broader than parameter-level continual learning. A candidate may change a prompt, memory schema, retrieval strategy, tool interface, workflow, routing rule, or orchestration policy. The same principles can be applied to model-weight changes, but those changes may require additional controls for training-data provenance, model integrity, supply-chain security, and compatibility. EGE does not specify a universal metric set; metric selection remains an application-specific modelling and governance decision.

This paper addresses the adoption decision through Evidence-Governed Evolution (EGE). EGE separates candidate generation from evidence-based adoption and combines: (i) fresh audit data collected after a candidate is frozen; (ii) meaningful improvement on at least one prespecified primary objective; (iii) noninferiority on every prespecified protected capability relative to a stable reference; (iv) mandatory governance checks; and (v) bounded canary deployment before persistent commitment.

The paper makes four bounded contributions. First, it formalises persistent adoption as a distinct decision problem for adaptively generated agent configurations, with separate development, calibration, confirmatoryaudit, and deployment stages. Second, it gives a simultaneous lower-confidence-bound construction that accounts for candidate-dependent evaluator discrepancy using independent calibration data rather than assuming a known calibration allowance. Third, it formalises reference replacement through bridge comparisons and derives a cumulative capability-degradation bound along the reference lineage; this result makes explicit the statistical price of changing the protected baseline. Fourth, it provides an audit-planning inequality and a reproducible Monte Carlo stress test that exposes a failure mode of evaluator-naive acceptance. The Monte Carlo experiment is not presented as evidence about a real agent system. A real-agent evaluation remains necessary to establish operational usefulness and is identified as an outstanding empirical requirement.

The distinction from prior work is deliberately narrow. EGE does not introduce a new concentration inequality, a new general-purpose sequential-testing principle, or a complete agent safety architecture. Its contribution is a specific composition and formalisation for persistent agent-update adoption: candidate-wise confirmatory testing, candidate-specific evaluator calibration, protected-capability reference lineage, and an operational commitment protocol. The novelty claim is therefore about this problem formulation and the resulting guarantees, not about the underlying Hoefding or union-bound tools.

The results are conditional statistical guarantees, not a proof of universal agent safety. In particular, rollback restores configuration but cannot reverse irreversible external actions.

## 2 Related Work

## 2.1 Reflection, Refinement, and Persistent Skills

Reflexion uses verbal feedback and episodic memory to improve subsequent agent attempts (Shinn et al., 2023). Self-Refine iteratively critiques and revises model outputs (Madaan et al., 2023). Voyager accumulates reusable skills through interaction with an environment (Wang et al., 2023). EGE is complementary: these methods can generate candidate changes, whereas EGE governs whether a candidate is retained persistently.

EvoPrompt connects language models with evolutionary algorithms for discrete prompt optimisation (Guo et al., 2024). Such optimisation makes the distinction between development evidence and final audit evidence especially important: the data used to select a candidate should not also be treated as an untouched confirmation sample.

Recent surveys organise self-evolving agents by the components and stages of evolution, examine selfimprovement in modern agentic systems, and study endogenous observations generated within agent interaction loops (Gao et al., 2026a; Ren et al., 2026; Zhang et al., 2026). These works motivate reliability concerns around self-generated learning signals; EGE focuses on the statistical and operational gate at adoption time.

## 2.2 Adaptive Data Analysis and Sequential Inference

Hoefding’s inequality gives finite-sample concentration bounds for independent bounded observations (Hoefding, 1963). Repeated adaptive reuse of a holdout can invalidate ordinary fixed-sample inference; reusableholdout methods address this problem through controlled information release (Dwork et al., 2015). Confidence sequences provide time-uniform inference when data are monitored repeatedly (Howard et al., 2021). Classical group-sequential designs and alpha-spending methods likewise address repeated interim testing (Lan and DeMets, 1983; DeMets and Lan, 1994). Conformal risk control ofers a complementary approach to controlling expected monotone losses under its own exchangeability and procedure-specific assumptions (Angelopoulos et al., 2024).

EGE’s simple construction is a union-bound allocation across candidate indices and metrics. It is valid when each candidate is frozen before a fresh audit sample is collected and the stated conditional sampling assumptions hold. It does not permit unrestricted reuse or repeated optional inspection of a fixed sample.

## 2.3 Safe Adaptation and Risk Constraints

Safe policy improvement studies evidence-supported policy changes relative to a baseline (Laroche et al., 2019). Broader safe reinforcement-learning research considers performance and safety constraints during learning and deployment (García and Fernández, 2015; Chow et al., 2018). EGE extends the governance perspective to heterogeneous agent configuration changes and multiple protected capabilities. Its non-inferiority tests do not imply that every possible failure mode is controlled.

## 2.4 Agent Self-Evolution Evaluation and the Adoption Gap

Recent work has begun to study self-evolving agents as a distinct research area. Gao et al. organise the field around what, when, and how an agent evolves, while Ren et al. survey self-improvement mechanisms in modern agentic systems (Gao et al., 2026a; Ren et al., 2026). Zhang et al. analyse endogenous observations as learning signals generated by an agent’s own generation, evaluation, and interaction processes (Zhang et al., 2026). Benchmarking work such as EvoAgentBench evaluates ability transfer during agent self-evolution (Gao et al., 2026b). Lin et al. distinguish the ability to produce useful harness updates from the ability of a task-solving agent to benefit from those updates, underscoring that update generation and update utility are separate capabilities (Lin et al., 2026). These studies strengthen the motivation for evaluating agent evolution directly. EGE addresses a diferent stage: after a candidate update has been generated, what evidence should be required before it is allowed to alter persistent system state?

Table 1: Conceptual positioning of EGE relative to representative research streams. Entries describe primary emphasis, not an exhaustive claim that a method cannot be combined with another control.
<table><tr><td>Research stream</td><td>Primary question</td><td>Relationship to EGE</td></tr><tr><td>Reflection and iterative re- finement</td><td>How can an agent improve a response or future attempt?</td><td>Can supply candidate modifications; does not by itself define persistent-adoption evidence.</td></tr><tr><td>Prompt and evolutionary optimisation</td><td>How can candidate prompts or configura- tions be searched?</td><td>Supplies candidate generation; develop- ment scores must not be reused as un- touched confirmation evidence.</td></tr><tr><td>Safe policy improvement</td><td>When is a policy change supported rela- tive to a baseline?</td><td>Provides related baseline-constrained rea- soning; EGE targets heterogeneous agent- configuration updates and multiple pro- tected capabilities.</td></tr><tr><td>and confidence sequences</td><td>Adaptive data analysis How can inference remain valid under re- peated or adaptive analysis?</td><td>Supplies statistical tools; EGE instanti- ates a candidate-wise fresh-audit design and explicit calibration allocation.</td></tr><tr><td>Agent self-evolution benchmarks</td><td>Do self-generated experiences or harness updates improve agent capability?</td><td>Supplies evaluation settings; EGE governs whether a particular update is persistently adopted.</td></tr><tr><td>release governance</td><td>Canary deployment and How can a change be exposed gradually and monitored?</td><td>Supplies operational containment; EGE separates this from offline statistical evi- dence.</td></tr></table>

The table is a conceptual map, not a systematic review or a claim of exhaustive coverage. In particular, safe policy improvement, adaptive inference, and deployment canary techniques are mature research areas. EGE must therefore be judged by whether the combination yields useful, correctly stated guarantees for this adoption problem, rather than by treating the combination of familiar components as novelty in itself.

## 3 Problem Formulation

## 3.1 Agent Configuration and Candidate Generation

Let $I _ { j }$ denote the incumbent configuration immediately before candidate index j is assessed. Its components are represented by

$$
I _ { j } = ( M _ { j } , P _ { j } , M _ { j } , T _ { j } , C _ { j } , G _ { j } ) ,\tag{1}
$$

where $M _ { j }$ is the model, $P _ { j }$ the prompt and policy configuration, $\mathcal { M } _ { j }$ persistent memory, $T _ { j }$ tools, $C _ { j }$ orchestration logic, and $G _ { j }$ governance constraints. A subset of these components may be immutable.

Given experience $\mathcal { E } _ { j }$ , a candidate generator produces

$$
\begin{array} { r } { \widetilde { A } _ { j } = \mathcal { U } ( I _ { j } , \mathcal { E } _ { j } ) . } \end{array}\tag{2}
$$

The generator may be an optimiser, an LLM-based editor, a search procedure, or a human-supervised process.   
EGE does not prescribe the generator.

## 3.2 Metrics and Paired Estimands

Let J be a nonempty finite set of primary objectives and K a finite set of protected capabilities. Their roles, score definitions, and evaluation distributions must be declared before audit outcomes are observed. Scores are normalised to [0, 1], so paired score diferences lie in [−1, 1]. The reference-evaluation procedure defines the target estimand used in the statistical guarantees; it need not be interpreted as error-free ground truth.

For metric $r ,$ let $q _ { r } ^ { R } ( A , Z )$ denote the reference-procedure score of configuration A on instance Z. Define

$$
D _ { j , r } ^ { R } ( Z ) = q _ { r } ^ { R } ( \widetilde { A } _ { j } , Z ) - q _ { r } ^ { R } ( B _ { j , r } , Z ) ,\tag{3}
$$

where

$$
B _ { j , r } = \left\{ \begin{array} { l l } { I _ { j } , } & { r \in \mathcal { I } , } \\ { A _ { \mathrm { r e f } } , } & { r \in \mathcal { K } . } \end{array} \right.
$$

Thus primary objectives are compared with the incumbent, while protected capabilities are compared with a fixed, versioned reference. If a metric belongs to both sets with diferent comparators or thresholds, the two roles are represented as distinct tests.

The target diference is

$$
\Delta _ { j , r } ^ { R } = \mathbb { E } _ { Z \sim \mathcal { D } _ { r } } [ D _ { j , r } ^ { R } ( Z ) \mid \mathcal { H } _ { j } ] ,\tag{4}
$$

where $\mathcal { H } _ { j }$ is the information available immediately before candidate $j$ is frozen and its fresh samples are drawn. The distribution $\mathcal { D } _ { r }$ is the prespecified target distribution for metric r. For a single configuration, define $Q _ { r } ^ { R } ( A ) = \mathbb { E } _ { Z \sim { \mathcal { D } } _ { r } } [ q _ { r } ^ { R } ( A , Z ) ]$ ; paired diferences are therefore diferences of these reference-procedure expectations when the same distribution is used.

## 3.3 Capability Representation

For exposition, an application may organise its evaluation metrics into a capability vector such as

$$
\mathbf { Q } ( A ; \mathcal { D } ) = \left[ Q _ { \mathrm { t a s k } } \quad Q _ { \mathrm { g e n } } \quad Q _ { \mathrm { r e t } } \quad Q _ { \mathrm { r e l } } \quad Q _ { \mathrm { e f f } } \quad Q _ { \mathrm { c a l } } \right] ^ { \mathsf { T } } ,\tag{5}
$$

where the entries represent task performance, generalisation, retention, reliability, eficiency, and calibration. This vector is illustrative rather than prescriptive. Each component must be translated into an operational metric with a defined scoring rule, task population, aggregation procedure, and direction of preference. For example, latency and cost are naturally minimisation metrics; they can be transformed into a higher-is-better score or handled by a separately specified upper confidence bound. The operational importance of these dimensions is also reflected in LLM-serving systems that optimise KV-cache memory use and the throughput– latency trade-of, including PagedAttention and Sarathi-Serve (Kwon et al., 2023; Agrawal et al., 2024). The present theory assumes higher scores are better and scores are bounded in [0, 1].

The distinction between primary and protected metrics is a policy choice made before candidate evaluation. A metric can be primary in one update campaign and protected in another. If a metric is assigned both roles for the same candidate but with diferent comparators, it should be represented as two tests rather than silently combining the roles. This avoids ambiguity in the number of tests and in the family-wise error allocation.

## 4 Evidence-Governed Adoption Rule

For each primary objective $r \in \mathcal { I }$ , let $\delta _ { r } > 0$ be the minimum practically meaningful improvement. For each protected capability $r \in \kappa$ , let $\epsilon _ { r } \geq 0$ be the maximum tolerated degradation. These margins must be fixed before audit evaluation.

The population-level acceptance requirements are

$$
\exists r \in \mathcal { I } : \quad \Delta _ { j , r } ^ { R } \geq \delta _ { r } ,\tag{6}
$$

$$
\forall r \in \mathcal { K } : \quad \Delta _ { j , r } ^ { R } \geq - \epsilon _ { r } ,\tag{7}
$$

$$
\forall m \in \mathcal { G } : \quad G _ { m } ( \widetilde { A } _ { j } ) = \mathrm { t r u e } ,\tag{8}
$$

where $\mathcal { G }$ is the set of mandatory governance checks. Examples include tool permissions, data-access restrictions, resource ceilings, and human-approval requirements. A high task score cannot compensate for failure of a mandatory constraint. Governance checks should be classified by enforcement mechanism. Deterministic checks can include allow-listed tool permissions, schema validation, network egress restrictions, resource ceilings, required approvals, and immutable audit logging. Human-review checks can cover changes with high impact or ambiguous consequences. Learned classifiers may assist triage, but should not be the sole enforcement mechanism for requirements whose violation is unacceptable. A check should specify its input, pass/fail semantics, version, owner, and failure action.

Definition 1 (Evidence-governed candidate). A candidate is evidence-governed if it is frozen before its audit sample is drawn, evaluated using fresh audit observations not used to construct or tune that candidate, and assessed using prespecified metrics, margins, and governance checks.

## 4.1 Evaluator Calibration

Let $q _ { r } ^ { E } ( A , Z )$ denote the operational evaluator score and $q _ { r } ^ { R } ( A , Z )$ the score produced by a prespecified reference-evaluation procedure. The reference procedure may be a deterministic verifier, an adjudicated human protocol, or another independently governed measurement procedure. The target estimand in this subsection is the mean paired diference under the reference procedure, not an unobservable notion of “true capability” unless the reference procedure is itself a valid ground-truth measurement.

For candidate $j ,$ , comparator $B _ { j , r }$ , and instance $Z ,$ define

$$
\begin{array} { r } { D _ { j , r } ^ { E } ( Z ) = q _ { r } ^ { E } ( \widetilde { A } _ { j } , Z ) - q _ { r } ^ { E } ( B _ { j , r } , Z ) , } \end{array}\tag{9}
$$

$$
D _ { j , r } ^ { R } ( Z ) = q _ { r } ^ { R } ( \widetilde { A } _ { j } , Z ) - q _ { r } ^ { R } ( B _ { j , r } , Z ) ,\tag{10}
$$

$$
W _ { j , r } ( Z ) = D _ { j , r } ^ { E } ( Z ) - D _ { j , r } ^ { R } ( Z ) .\tag{11}
$$

Scores are in [0, 1], so $D ^ { E } , D ^ { R } \in [ - 1 , 1 ]$ and $W \in [ - 2 , 2 ]$ . Write $\mu _ { j , r } ^ { E } = \mathbb { E } [ D _ { j , r } ^ { E } \mid \mathcal { H } _ { j } ] , \ : \Delta _ { j , r } ^ { R } = \mathbb { E } [ D _ { j , r } ^ { R } \mid \mathcal { H } _ { j } ]$ , and $b _ { j , r } = \operatorname { \mathbb { E } } [ W _ { j , r } \mid \mathcal { H } _ { j } ]$ . Then $\mu _ { j , r } ^ { E } - \Delta _ { j , r } ^ { R } = b _ { j , \ i }$ <sub>r</sub>.

Unlike a fixed, unexplained allowance, EGE estimates the discrepancy for each frozen candidate using an independent calibration sample drawn from the same declared metric distribution. This is intentionally demanding: a candidate-specific calibration set is needed because an update may change output style or behaviour in ways that afect evaluator agreement. A global calibration result can replace this design only if a separate uniform-generalisation argument covers the complete candidate class and adaptive selection process.

The calibration procedure does not eliminate construct-validity concerns. If the reference evaluator is systematically wrong, the resulting guarantee is only relative to that reference procedure. If human adjudication is used, the paper’s target is the expected adjudicated score under the stated protocol; inter-rater disagreement and adjudication reliability must be reported. The reference procedure, rubric, blinding, version, and sampling frame must be fixed before confirmatory results are examined.

## 4.2 Evidence Partitioning and Audit Integrity

A practical implementation should distinguish at least three information partitions. The development partition is available to the candidate generator and may be used for prompt iteration, debugging, reflection, and search. The validation partition may be used to compare alternative candidate designs during development, but repeated adaptive use means it should not be treated as a final confirmatory sample. The audit partition is reserved for the frozen candidate and is revealed only after the candidate, comparator, metrics, margins, and analysis procedure have been committed.

The audit partition should be controlled as an operational resource. Access logs should record which candidate was evaluated, which audit instances were used, the evaluator version, the scoring rubric, and whether any audit result was exposed to the candidate generator. If an audit result is used to modify the candidate, that sample is no longer fresh for the modified candidate. The modified candidate must receive a new confirmatory sample or be evaluated with a method that explicitly accounts for adaptive reuse. This requirement is stronger than merely labelling a dataset as a test set: the information flow from audit results back into candidate construction must also be controlled.

Where tasks are clustered by user, session, domain, or environment, sampling individual rows independently may be inappropriate. The unit of sampling should reflect the dependence structure. For example, if several test cases arise from one user session, treating them as independent can understate uncertainty. A cluster-level design, a concentration inequality for dependent observations, or an appropriately conservative aggregation is then required. The Hoefding result below is not automatically valid for arbitrary correlated task traces.

## 5 Sequential Statistical Adoption

## 5.1 Sampling and Calibration Assumptions

For each candidate j and metric r, collect ${ n _ { j , \prime } }$ fresh confirmatory observations $d _ { i , j , r } ^ { E }$ and $m _ { j , r }$ independent calibration observations $w _ { i , j , r }$ . Define their means by

$$
\bar { d } _ { j , r } ^ { E } = \frac { 1 } { n _ { j , r } } \sum _ { i = 1 } ^ { n _ { j , r } } d _ { i , j , r } ^ { E } ,\tag{12}
$$

$$
\bar { w } _ { j , r } = \frac { 1 } { m _ { j , r } } \sum _ { i = 1 } ^ { m _ { j , r } } w _ { i , j , r } .\tag{13}
$$

Assumption 1 (Conditional fresh sampling). Conditional on the pre-audit history $\mathcal { H } _ { j }$ , candidate and comparator are frozen before either sample is drawn. Confirmatory observations are independent, lie in $[ - 1 , 1 ]$ and have common conditional mean $\mu _ { j , r } ^ { \hat { E } }$ . Calibration observations are independent, lie in $[ - 2 , 2 ]$ , and have common conditional mean $b _ { j , r }$ . The two samples are mutually independent conditional on $\mathcal { H } _ { j }$ , and neither is used to construct or tune the frozen candidate. Samples are drawn from the declared distribution for metric r.

Assumption 2 (Finite prespecified metric family). The primary and protected test sets are fixed and finite. Let $R = | \mathcal { I } | + | \mathcal { K } |$ , counting a metric twice if it appears in both roles with diferent comparators or thresholds. Metric definitions, score ranges, distributions, margins, and reference-evaluation procedures are fixed before confirmatory outcomes are observed.

Assumption 3 (Reference-procedure validity). For each test, the reference evaluator defines the target estimand $\Delta _ { j , r } ^ { R } .$ Any claim that this estimand represents an external capability relies on the validity of the reference procedure and its sampling frame. The theorem below does not establish that construct-validity claim.

## 5.2 Evaluation and Calibration Bounds

For an evaluation error allocation $e _ { j , r } \in ( 0 , 1 )$ , Hoefding’s inequality gives

$$
L _ { j , r } ^ { E } = \bar { d } _ { j , r } ^ { E } - \sqrt { \frac { 2 \ln ( 1 / e _ { j , r } ) } { n _ { j , r } } } ,\tag{14}
$$

with $\mathrm { P r } ( L _ { j , r } ^ { E } > \mu _ { j , r } ^ { E } \mid \mathcal { H } _ { j } ) \leq e _ { j , r }$

For a calibration error allocation $c _ { j , r } \in ( 0 , 1 )$ , the one-sided Hoefding inequality for observations in an interval of width four gives

$$
U _ { j , r } ^ { b } = \bar { w } _ { j , r } + \sqrt { \frac { 8 \ln ( 1 / c _ { j , r } ) } { m _ { j , r } } } ,\tag{15}
$$

with $\mathrm { P r } ( b _ { j , r } > U _ { j , r } ^ { b } \mid \mathcal { H } _ { j } ) \leq c _ { j , r }$

Since $\Delta _ { j , r } ^ { R } = \mu _ { j , r } ^ { E } - b _ { j , r } , :$ a lower confidence bound for the reference-procedure estimand is

$$
L _ { j , r } ^ { R } = L _ { j , r } ^ { E } - U _ { j , r } ^ { b } .\tag{16}
$$

On the joint event $L _ { j , r } ^ { E } \leq \mu _ { j , r } ^ { E }$ and $b _ { j , r } \leq U _ { j , r } ^ { b }$ , we have

$$
L _ { j , r } ^ { R } \leq \mu _ { j , r } ^ { E } - U _ { j , r } ^ { b } \leq \mu _ { j , r } ^ { E } - b _ { j , r } = \Delta _ { j , r } ^ { R } .
$$

The calibration bound is one-sided because only optimistic evaluator discrepancy can make an evaluator-based lower bound overstate the reference-procedure efect. Negative discrepancy is not subtracted as if it were harmful bias; the resulting lower bound remains valid for the signed target estimand.

## 5.3 Error Allocation Across Candidates and Tests

Let $\alpha \in ( 0 , 1 )$ be the global error budget. For global audit-event index $j = 1 , 2 , . . .$ . and test $r = 1 , \ldots , R$ allocate

$$
e _ { j , r } = c _ { j , r } = \frac { 3 \alpha } { \pi ^ { 2 } j ^ { 2 } R } .\tag{17}
$$

Because $\textstyle \sum _ { j \geq 1 } j ^ { - 2 } = \pi ^ { 2 } / 6$ , the evaluation allocations sum to $\alpha / 2$ and the calibration allocations also sum to $\alpha / 2$ . Therefore,

$$
\sum _ { j = 1 } ^ { \infty } \sum _ { r = 1 } ^ { R } ( e _ { j , r } + c _ { j , r } ) = \alpha .\tag{18}
$$

Theorem 1 (Simultaneous evaluator-calibrated coverage). Under Assumptions $^ { 1 - 3 , }$ with the allocations in $( 1 \gamma )$

$$
\operatorname* { P r } \left\{ L _ { j , r } ^ { R } \leq \Delta _ { j , r } ^ { R } \ f o r \ e v e r y \ j \geq 1 , \ r \in \left\{ 1 , \ldots , R \right\} \right\} \geq 1 - \alpha .\tag{19}
$$

No independence across candidate indices or metric tests is required for the final union-bound step.

Proof. For each fixed $( j , r )$ , Assumption 1 and Hoefding’s inequality imply that the evaluation-bound failure probability is at most $e _ { j , r }$ and the calibration-bound failure probability is at most $c _ { j , r } .$ . By the deterministic implication preceding the theorem, failure of $L _ { j , r } ^ { R } \leq \Delta _ { j , r } ^ { R }$ is contained in the union of those two events. Hence

$$
\mathrm { P r } \{ L _ { j , r } ^ { R } > \Delta _ { j , r } ^ { R } \} \leq e _ { j , r } + c _ { j , r } .
$$

A union bound over the countable candidate sequence and finite metric family yields

$$
\operatorname* { P r } \{ \exists j , r : L _ { j , r } ^ { R } > \Delta _ { j , r } ^ { R } \} \leq \sum _ { j = 1 } ^ { \infty } \sum _ { r = 1 } ^ { R } ( e _ { j , r } + c _ { j , r } ) = \alpha .
$$

This proves the result. Adaptivity of candidate generation is permitted because the candidate is fixed conditional on $\mathcal { H } _ { j }$ before its independent fresh evaluation and calibration samples are drawn. The index is a global audit-event index, allocated by the audit controller to candidate tests and reference-bridge tests alike; it cannot be reset after rejection, rollback, or a change in proposal strategy. □

Remark 1 (What the theorem does not establish). The theorem controls simultaneous statistical undercoverage relative to the reference-evaluation estimand. It does not prove that the reference evaluator measures the intended real-world capability, that the audit distribution matches deployment, or that an agent cannot cause harm through unmeasured actions. Those are separate validity and governance questions.

## 5.4 Audit Planning and a Conservative Power Condition

Define the evaluation and calibration radii

$$
h _ { j , r } ^ { E } = \sqrt { \frac { 2 \ln ( 1 / e _ { j , r } ) } { n _ { j , r } } } , \qquad h _ { j , r } ^ { b } = \sqrt { \frac { 8 \ln ( 1 / c _ { j , r } ) } { m _ { j , r } } } .
$$

For a test with threshold $\tau _ { r }$ and true reference-procedure efect $\Delta _ { j , r } ^ { R } = \tau _ { r } + g$ for $g > 0$ , the event $\bar { d } ^ { E } \geq \mu ^ { E } - h ^ { E }$ and $\bar { w } - b \leq h ^ { b }$ implies

$$
L _ { j , r } ^ { R } \geq \Delta _ { j , r } ^ { R } - h _ { j , r } ^ { E } - 2 h _ { j , r } ^ { b } .
$$

Consequently, on the joint event $\bar { d } _ { j , r } ^ { E } \geq \mu _ { j , r } ^ { E } - h _ { j , r } ^ { E }$ and $\bar { w } _ { j , r } - b _ { j , r } \leq h _ { j , r } ^ { b }$ , the constructed lower bound obeys

$$
L _ { j , r } ^ { R } \geq \Delta _ { j , r } ^ { R } - 2 h _ { j , r } ^ { E } - 2 h _ { j , r } ^ { b } .
$$

The factor of two on the evaluation radius is necessary because the lower confidence bound itself subtracts one radius from the observed mean, while the event used to lower-bound that observed mean contributes another radius. Therefore, if $2 h _ { j , r } ^ { E } + 2 h _ { j , r } ^ { b } \leq g .$ , the test passes with probability at least $\displaystyle 1 - e _ { j , r } - c _ { j , r }$ . This is a suficient, conservative power condition, not a necessary condition.

For any split parameter $\lambda \in ( 0 , 1 )$ , it is suficient to choose

$$
n _ { j , r } \geq \frac { 8 \ln ( 1 / e _ { j , r } ) } { \lambda ^ { 2 } g ^ { 2 } } ,\tag{20}
$$

$$
m _ { j , r } \geq \frac { 3 2 \ln ( 1 / c _ { j , r } ) } { ( 1 - \lambda ) ^ { 2 } g ^ { 2 } } .\tag{21}
$$

The parameter λ allocates the available margin between confirmatory evaluation and calibration uncertainty. The evaluation bound above is four times the fixed-radius requirement in (24) because the power argument must reserve margin both for the LCB construction and for downward sampling fluctuation. These bounds expose the cost of candidate-wise calibration: the required calibration sample can be substantial when the desired margin is small or the candidate index is large. They also show why a claim of practical eficiency requires empirical cost reporting rather than only a validity theorem.

## 5.5 Statistical Acceptance Predicate

The candidate passes the statistical gate if

$$
\begin{array} { l } { \displaystyle \mathrm { A c c e p t } _ { { \boldsymbol j } } ^ { \mathrm { s t a t } } = \left[ \exists { \boldsymbol r } \in { \boldsymbol J } : L _ { { \boldsymbol j } , { \boldsymbol r } } ^ { R } \geq \delta _ { { \boldsymbol r } } \right] } \\ { \displaystyle \wedge \left[ \forall { \boldsymbol r } \in { \boldsymbol K } : L _ { { \boldsymbol j } , { \boldsymbol r } } ^ { R } \geq - \epsilon _ { { \boldsymbol r } } \right] . } \end{array}\tag{22}
$$

The lower-bound rule is intentionally conservative. A candidate does not pass merely because its observed mean is positive; the lower confidence bound must exceed the practical improvement margin. This distinguishes statistical evidence from efect-size relevance. A very small positive efect can be statistically distinguishable from zero yet still be operationally unimportant. Conversely, a candidate with a large observed efect but a small audit sample may remain inconclusive.

The adjusted lower bound can be read as the operational-evaluator sample mean minus the evaluation radius and the one-sided calibration upper bound. A primary test passes when $\bar { d } _ { j , r } ^ { E } - h _ { j , r } ^ { E } - U _ { j , r } ^ { b } \geq \delta _ { r }$ . A protected test passes when $\bar { d } _ { j , r } ^ { E } - h _ { j , r } ^ { E } - U _ { j , r } ^ { b } \geq - \epsilon _ { r }$ . This decomposition is useful for audit planning because it shows whether sampling uncertainty or evaluator discrepancy dominates an inconclusive decision.

## 5.6 Interpretation of Inconclusive Outcomes

The framework has three operationally distinct outcomes. A candidate can be accepted when every required statistical and governance condition passes. It can be rejected when a hard constraint fails or the evidence establishes that a required condition is not met. It can also remain inconclusive when the lower bound does not pass a threshold but the data do not establish a material regression. In a conservative deployment policy, both rejection and inconclusive outcomes retain the incumbent; however, the distinction should be preserved in logs because it matters for subsequent candidate development and audit-cost analysis. Additional sampling is permissible only if the sampling plan was prespecified or if the monitoring procedure is sequentially valid.

The complete ofline acceptance predicate is

$$
\mathrm { A c c e p t } _ { j } = \mathrm { A c c e p t } _ { j } ^ { \mathrm { s t a t } } \wedge \bigwedge _ { m \in \mathcal { G } } \{ G _ { m } ( \widetilde { A } _ { j } ) = \mathrm { t r u e } \} .\tag{23}
$$

Corollary 1 (Family-wise control of false statistical acceptance). Under the theorem’s assumptions and the evaluator-calibration procedure, the probability that any candidate in the audit campaign is statistically accepted while failing at least one population requirement in $( 6 ) \_ ( 7 )$ is at most α.

Proof. On the simultaneous-coverage event, every adjusted lower bound is no greater than its target population diference. If all primary objectives fail their respective meaningful-improvement thresholds, no primary lower bound can pass. If any protected capability violates its non-inferiority margin, its lower bound cannot pass. Thus a population-invalid candidate cannot pass the statistical gate on the simultaneous-coverage event. The complement has probability at most α. □

## 6 Reference-Anchored Capability Stability and Reference Lineage

A fixed reference prevents the protected baseline from moving after every accepted update. If the reference is eventually replaced, however, a new baseline can reintroduce cumulative drift unless the replacement is explicitly compared with its predecessor. EGE therefore treats reference replacement as a separately audited governance event.

Proposition 1 (Fixed-reference non-inferiority). On the simultaneous-coverage event in Theorem 1, every candidate that passes all protected-capability tests in a fixed-reference epoch satisfies

$$
\Delta _ { j , r } ^ { R } ( \widetilde { A } _ { j } , A _ { \mathrm { r e f } } ) \geq - \epsilon _ { r } , \qquad r \in { \mathcal { K } } .
$$

Proof. For every protected test, simultaneous coverage gives $L _ { j , r } ^ { R } \leq \Delta _ { j , r } ^ { R }$ . Acceptance requires $L _ { j , r } ^ { R } \ge - \epsilon _ { r }$ Combining the inequalities proves the claim. □

## 6.1 Reference Replacement and Bridge Tests

Let $A ^ { ( 0 ) } , A ^ { ( 1 ) } , \ldots , A ^ { ( K ) }$ denote a sequence of approved references. At transition $k ,$ a bridge test compares the proposed new reference $A ^ { ( k + 1 ) }$ with the previous reference $A ^ { ( k ) }$ on the same protected metric and target distribution. Let its target diference be

$$
B _ { k , r } = Q _ { r } ^ { R } ( A ^ { ( k + 1 ) } ) - Q _ { r } ^ { R } ( A ^ { ( k ) } ) .
$$

The bridge test must use fresh evaluation and calibration samples and receive its own error allocation within the global budget. In implementation, every candidate test and every bridge test receives a unique, non-reusable index from one global audit controller; the summable allocation in (17) is applied to this global event index. Reference transitions therefore cannot reuse error budget already spent on candidate tests. Let $L _ { k , r } ^ { \mathrm { b r i d g e } }$ be its evaluator-calibrated lower confidence bound. The replacement is permitted only if $L _ { k , r } ^ { \mathrm { b r i d g e } } \geq - \rho _ { k , r }$ , where $\rho _ { k , r } \geq 0$ is a prespecified bridge tolerance. A reference change is therefore not an administrative reset: it consumes statistical evidence and is recorded as a distinct lineage transition.

Theorem 2 (Cumulative capability bound along a reference lineage). Assume all candidate tests and all bridge tests are included in one simultaneous-coverage family with total error budget α. Assume every committed configuration A in epoch K passes its protected-capability test relative to $A ^ { ( K ) }$ with tolerance $\epsilon _ { K , r } ,$ and every accepted reference transition satisfies $B _ { k , r } \geq - \rho _ { k , r }$ . Then, on the simultaneous-coverage event,

$$
Q _ { r } ^ { R } ( A ) - Q _ { r } ^ { R } ( A ^ { ( 0 ) } ) \geq - \left( \epsilon _ { K , r } + \sum _ { k = 0 } ^ { K - 1 } \rho _ { k , r } \right) .
$$

Thus the maximum allowed degradation relative to the original reference is explicit and accumulates only through declared epoch tolerances.

Proof. By the candidate’s protected-capability acceptance condition and simultaneous coverage, $Q _ { r } ^ { R } ( A ) -$ $Q _ { r } ^ { R } ( A ^ { ( K ) } ) \geq - \epsilon _ { K , r }$ . For each bridge transition, simultaneous coverage and its acceptance condition give ${ Q } _ { r } ^ { R } ( A ^ { ( k + 1 ) } ) - { Q } _ { r } ^ { R } ( A ^ { ( k ) } ) \geq - \rho _ { k , r }$ . Add these inequalities and telescope the intermediate reference terms:

$$
Q _ { r } ^ { R } ( A ) - Q _ { r } ^ { R } ( A ^ { ( 0 ) } ) = Q _ { r } ^ { R } ( A ) - Q _ { r } ^ { R } ( A ^ { ( K ) } ) + \sum _ { k = 0 } ^ { K - 1 } \left[ Q _ { r } ^ { R } ( A ^ { ( k + 1 ) } ) - Q _ { r } ^ { R } ( A ^ { ( k ) } ) \right] .
$$

The claimed lower bound follows.

The result requires the same metric definition and target distribution across all bridge comparisons. If the target distribution changes, the terms are not directly additive without an additional transport or standardisation argument. The bound is also only as meaningful as the reference-evaluation procedure. It formalises reference lineage; it does not prove that the initial reference is optimal or that the selected protected metrics exhaust all relevant capabilities

## 7 Evidence-Governed Deployment

Ofline statistical acceptance does not imply safe behaviour under production load or on unrepresented tasks.   
EGE therefore separates statistical acceptance from persistent commitment.

## 7.1 Two-Phase Commitment

The operational state machine has four states: (1) Candidate, where a proposed modification is frozen; (2) Audit-passed, where statistical and governance requirements pass; (3) Canary, where the candidate runs under bounded trafic, permissions, workload, or execution scope; and (4) Committed, where it becomes the persistent configuration after prespecified canary criteria pass.

A candidate failing ofline evaluation is rejected. A candidate failing canary criteria is contained and the incumbent configuration is restored. Configuration rollback does not reverse irreversible external side efects; systems with consequential tool access need transaction controls and compensating-action procedures.

The canary scope can be limited along several dimensions: the fraction of eligible requests routed to the candidate, the set of users or tenants exposed, the classes of tools enabled, the maximum number of tool calls, the permitted data domains, the execution time, and the maximum resource budget. The appropriate controls depend on the agent’s action space. For a read-only summarisation agent, a trafic-split canary may be suficient. For an agent that can modify records or trigger external workflows, a shadow mode or approval-gated mode may be more appropriate than live execution.

The canary must have an explicit stop condition and an explicit authority to stop. A useful operational record includes the candidate hash, incumbent hash, canary start and end times, exposure volume, metric values, alert events, rollback reason, and any compensating actions. The candidate should not be permitted to change itself during the canary; otherwise, the object being monitored is no longer the object that passed the ofline audit. If adaptation during canary operation is required, each resulting version should be treated as a new candidate or covered by a separately justified online-learning protocol.

## 7.2 Operational Monitoring

Canary criteria should be application-specific and fixed before deployment. They may include task success, tool-error rate, latency, resource consumption, policy violations, escalation rate, and user-impact indicators. Repeated monitoring requires sequentially valid procedures or deterministic operational limits; a fixed-sample confidence interval must not be repeatedly inspected as if it were time-uniform.

## 7.3 Risk-Tiered Deployment

A practical deployment controller can assign candidate changes to risk tiers before evaluation. A low-impact change might alter wording or reorder a non-critical retrieval preference while preserving permissions and tool interfaces. A medium-impact change might modify memory retrieval, task routing, or multi-step workflow logic. A high-impact change might modify access to sensitive data, external-action permissions, financial or operational workflows, or safety-relevant control logic. These examples are illustrative; the organisation must define its own classification.

The risk tier should determine the required canary scope, duration, approval level, and rollback authority. It should not change the statistical meaning of the audit results after they have been observed. Higher-risk changes may require stricter margins, more protected metrics, deterministic checks, or mandatory human approval. Such requirements should be encoded before the candidate is evaluated. This preserves the separation between evidence and policy while allowing governance controls to scale with consequence.

![](images/8be640ac363387328a995978f4382cf9ff08c6269086b6e3b0c1cf873513bde3.jpg)  
Figure 1: Evidence-Governed Evolution separates candidate generation, fresh statistical evaluation, mandatory governance checks, and bounded deployment. Ofline failure leads to rejection; only a candidate that passes both ofline checks and canary monitoring is persistently committed.

Algorithm 1 Evidence-Governed Evolution   
Require: Current incumbent ${ \overline { { I _ { j } } } } ,$ reference $A _ { \mathrm { { r e f } } } ,$ experience $\overline { { \mathcal { E } _ { j } } }$   
Require: Margins $\delta , \epsilon ,$ error budget α, metric count R   
1: Generate $\widetilde { A _ { j } }  \mathcal { U } ( I _ { j } , \mathcal { E } _ { j } )$   
2: Freeze candidate; do not tune it using confirmatory audit outcomes   
3: For each metric $r ,$ draw independent fresh confirmatory and calibration samples   
4: Compute $e _ { j , r } , c _ { j , r }$ using (17)   
5: Compute evaluator LCB and calibration UCB using (14)–(15)   
6: Compute reference-procedure LCB $L _ { j , r } ^ { R }$ using (16)   
7: Evaluate primary and protected-capability tests in (22)   
8: Evaluate all mandatory governance checks   
9: if any required condition fails then   
10: return Reject candidate; retain $I _ { j }$   
11: end if   
12: Deploy candidate in bounded canary scope   
13: if all prespecified canary criteria pass then   
14: $I _ { j + 1 }  \widetilde { A } _ { j }$   
15: return Commit $I _ { j + 1 }$   
16: else   
17: Contain candidate; restore $I _ { j }$   
18: Execute compensating actions where applicable   
19: return Roll back; retain $I _ { j }$   
20: end if

## 7.4 Rollback and Recovery

Rollback should be implemented as a tested recovery procedure rather than an informal instruction to restore a prior prompt. A complete recovery package may include the prior configuration, memory-schema compatibility information, tool and API versions, feature flags, routing state, and any migration steps required to restore service. If a candidate changes persistent memory, rollback may require restoring a snapshot or applying a reversible migration. If it writes to external systems, recovery may require compensating transactions and an incident process. Some side efects cannot be reversed; the system should identify those actions before enabling the candidate and apply stricter controls accordingly.

Table 2: Monte Carlo sensitivity to evaluator optimism and sample size (3,000 campaigns per condition; 10 candidates per campaign).
<table><tr><td>n = m</td><td>Bias b</td><td>EGE harmful campaigns</td><td>Naive harmful campaigns</td><td>EGE power</td><td>Naive power</td></tr><tr><td>5,000</td><td>0.00</td><td>0/3,000</td><td>0/3,000</td><td>52.6%</td><td>100.0%</td></tr><tr><td>5,000</td><td>0.10</td><td>0/3,000</td><td>0/3,000</td><td>76.2%</td><td>100.0%</td></tr><tr><td>5,000</td><td>0.25</td><td>0/3,000</td><td>2,997/3,000</td><td>65.7%</td><td>100.0%</td></tr><tr><td>20,000</td><td>0.00</td><td>0/3,000</td><td>0/3,000</td><td>100.0%</td><td>100.0%</td></tr><tr><td>20,000</td><td>0.10</td><td>0/3,000</td><td>0/3,000</td><td>100.0%</td><td>100.0%</td></tr><tr><td>20,000</td><td>0.25</td><td>0/3,000</td><td>2,997/3,000</td><td>100.0%</td><td>100.0%</td></tr><tr><td>80,000</td><td>0.00</td><td>0/3,000</td><td>0/3,000</td><td>100.0%</td><td>100.0%</td></tr><tr><td>80,000</td><td>0.10</td><td>0/3,000</td><td>0/3,000</td><td>100.0%</td><td>100.0%</td></tr><tr><td>80,000</td><td>0.25</td><td>0/3,000</td><td>2,997/3,000</td><td>100.0%</td><td>100.0%</td></tr></table>

## 8 Evaluation

This section reports one executed, fully specified Monte Carlo stress test and defines the additional real-agent evaluation required to assess operational usefulness. The Monte Carlo study is synthetic by construction: it tests a statistical failure mode under known data-generating assumptions. It is not evidence that EGE improves the behaviour of an implemented autonomous agent.

## 8.1 Monte Carlo sensitivity study: evaluator bias and audit size

Purpose and design. We ran a targeted Monte Carlo sensitivity study to examine two distinct properties of the proposed gate: (i) whether candidate-specific calibration prevents harmful acceptance when the operational evaluator is optimistic, and (ii) how much acceptance power is lost when calibration and audit samples are small. This is a synthetic statistical experiment, not an autonomous-agent benchmark. Each condition comprises 3,000 independent campaigns with 10 candidates per campaign. Each candidate is beneficial or harmful with probability one half. The reference-procedure primary efect is +0.25 for beneficial candidates and −0.10 for harmful candidates; the protected efect is zero. The primary threshold is $\delta = 0 . 0 5$ , the protected non-inferiority margin is $\epsilon = 0 . 2 0$ , and the global error budget is $\alpha = 0 . 0 5$ . The operational evaluator’s primary-score optimism is varied over $b \in \{ 0 , 0 . 1 0 , 0 . 2 5 \}$ ; protected-metric bias is zero. Confirmatory paired diferences are sampled from $\{ - 1 , + 1 \}$ with the specified means, while calibration discrepancies are sampled from $\{ - 2 , + 2 \}$ with the specified mean. The confirmatory and calibration samples are independent. We vary both sample sizes together over $n = m \in \{ 5 , 0 0 0 , 2 0 , 0 0 0 , 8 0 , 0 0 0 \}$ per candidate and metric. The candidate-wise evaluation and calibration allocations are each $3 \alpha / ( \pi ^ { 2 } j ^ { 2 } R )$ , so the two families jointly consume the global error budget. The random seed is 20261112.

Comparators and outcomes. The calibrated EGE comparator uses the evaluation lower confidence bound and the independent one-sided calibration upper confidence bound. The evaluator-naive comparator uses the same evaluation lower confidence bound but omits calibration. A campaign-level harmful-acceptance event occurs if at least one harmful candidate passes the primary and protected tests. Beneficial-candidate acceptance is the proportion of beneficial candidates accepted. The table reports observed campaign rates and beneficial-candidate acceptance frequencies. For zero observed campaign events, the one-sided exact 95% binomial upper confidence limit is reported; for nonzero event rates, two-sided exact 95% Clopper–Pearson intervals are used in the supplementary JSON.

For the calibrated rule, no harmful campaign acceptance was observed in any of the nine correctly specified conditions. The corresponding one-sided exact 95% upper confidence limit is approximately $9 . 9 8 \times 1 0 ^ { - 4 }$ for each condition; this Monte Carlo limit is not the theorem’s α guarantee. Under $b = 0 . 2 5$ , the evaluator-naive rule accepted at least one harmful candidate in 2,997 of 3,000 campaigns (99.9%). Under b = 0 and $b = 0 . 1 0$ it recorded no harmful campaign acceptance in this particular design. The contrast is therefore specific to the selected bias/efect combinations and should not be read as a general ranking of methods under all evaluator regimes.

Table 3: Calibration-distribution mismatch stress test (assumption deliberately violated).
<table><tr><td></td><td></td><td>Actual audit bias Calibration bias Harmful-acceptance campaigns</td><td>Observed rate</td></tr><tr><td>+0.25</td><td>+0.05</td><td>3,000/3,000</td><td>100.0%</td></tr></table>

Table 4: Illustrative audit and calibration sample-size requirements under the corrected suficient-power condition.
<table><tr><td>Global event index j</td><td>Per-test e = c</td><td>n minimum</td><td>m minimum</td><td>Total</td></tr><tr><td>1</td><td> $5 . 0 7 \times 1 0 ^ { - 3 }$ </td><td>16,913</td><td>67,651</td><td>84,564</td></tr><tr><td>10</td><td> $5 . 0 7 \times 1 0 ^ { - 5 }$ </td><td>31,650</td><td>126,597</td><td>158,247</td></tr><tr><td>100</td><td> $5 . 0 7 \times 1 0 ^ { - 7 }$ </td><td>46,386</td><td>185,543</td><td>231,929</td></tr></table>

The calibrated method’s beneficial-candidate acceptance frequency was 52.6–76.2% at 5,000 observations per sample, then 100% at 20,000 and 80,000 observations. This illustrates a real trade-of: calibration can block optimistic false gains but can also make the gate inconclusive when audit resources are insuficient. The 100% frequencies arise from deliberately separated efects and are not a universal power guarantee. Full campaign-level counts, confidence intervals, and seed are included in the reproducibility files.

## 8.2 Calibration-distribution mismatch stress test

The guarantee requires the calibration and confirmatory samples to represent the same declared metric distribution. To illustrate why this assumption matters, we ran a deliberately misspecified condition with 3,000 campaigns, 10 harmful candidates per campaign, $n = m = 1 0 0 , 0 0 0$ , reference-procedure primary efect −0.10, actual audit-evaluator optimism +0.25, but calibration-sample optimism only +0.05. All other settings and the nominal allocation were unchanged. The calibrated procedure accepted at least one harmful candidate in all 3,000 campaigns (100%). This is not a counterexample to the theorem: the calibration sample is not drawn from the same discrepancy distribution as the audit sample, so the stated assumptions are violated. It is a diagnostic showing that a formal confidence bound cannot compensate for a nonrepresentative calibration design.

The exact two-sided 95% Clopper–Pearson interval for the observed 3,000/3,000 rate is approximately [0.9988, 1.0000]. This interval quantifies Monte Carlo uncertainty only; it does not turn the misspecified experiment into a valid test of EGE’s theorem.

The planning table uses $\alpha = 0 . 0 5 , R = 3$ , target gap $g = 0 . 1 0$ , and split parameter $\lambda = 0 . 5$ . Counts are rounded upward from the suficient conditions in Section 5.4; they are per metric test and per candidate/bridge event. The calibration requirement is larger because the discrepancy variable has range width four, compared with width two for the paired evaluation diference. The growth with j reflects the summable allocation and makes the audit-cost implications explicit.

## 8.3 Required Real-Agent Evaluation Protocol

A real-agent study remains necessary before submission to a venue expecting evidence of practical utility. The study should use a reproducible agent whose editable harness is limited to declared components such as prompts, retrieval settings, tool-routing rules, or workflow configuration. Candidate generation must be separated from audit and reference-evaluation services. Each candidate should be frozen and content-hashed before independent audit and calibration samples are drawn.

Tasks and metrics. Use a task suite with deterministic verifiers wherever possible; benchmark design and red-teaming studies can inform breadth and adversarial coverage (Hendrycks et al., 2021; Perez et al., 2022). Include at least one primary objective and multiple protected capabilities chosen before the experiment. Candidate and comparator should be evaluated on the same task instances when pairing is valid. For stochastic agents, record model version, decoding settings, tool versions, seeds, timeouts, and retry rules. If tasks are clustered by user/session or share state, use cluster-level sampling or a concentration result appropriate to the dependence structure. Do not apply the independent-observation Hoefding bound to correlated traces without justification.

Candidate generation and baselines. Use the same candidate proposal stream for all adoption strategies where feasible. Compare (i) greedy observed-score acceptance, (ii) fixed observed-mean thresholds, (iii) confidence-bound acceptance without evaluator calibration, (iv) calibrated EGE without reference protection, (v) calibrated EGE with incumbent-only protection, and (vi) full EGE. Report candidate counts, accepted candidates, harmful acceptances, false rejections, inconclusive decisions, audit observations, calibration observations, elapsed time, and canary exposure.

Ablations and stress conditions. Vary candidate index, audit size, calibration size, evaluator optimism, evaluator drift, protected-capability regression, and task-distribution shift. Include an evaluator-manipulation condition in which candidate changes alter output style or structure while leaving task success unchanged. Report evaluator/reference disagreement by candidate class and style. Use blinded human adjudication or deterministic verifiers as the reference procedure; if human ratings are used, report the rubric, rater training, agreement, adjudication, and sampling frame. The reference procedure must not be described as ground truth unless that claim is justified.

Statistical reporting. Report campaign-level false-acceptance rates with exact or otherwise justified uncertainty intervals, acceptance power under prespecified efect sizes, per-capability regression rates, and audit cost. Report simulation seeds and number of repetitions. For agent experiments, report uncertainty across independent update episodes and avoid treating multiple tasks from one shared history as independent episodes. Separate ofline false acceptance from canary failure: the former is controlled only under the theorem’s assumptions, whereas the latter depends on deployment representativeness and monitoring design.

Canary study. A candidate that passes ofline evaluation should enter a bounded, immutable canary. Predeclare exposure limits, permitted tools, monitoring metrics, stop rules, and rollback authority. If canary metrics are inspected repeatedly, use a sequentially valid procedure or deterministic stop thresholds; do not repeatedly inspect a fixed-sample interval as if it were time-uniform. Record irreversible side efects and compensating actions. The canary stage must not be described as statistically guaranteed by the ofline theorem.

## 8.4 Reproducibility and Availability

The supplementary simulation script specifies the random seed, data-generating process, allocation, sample sizes, thresholds, and outcome calculations. A real-agent reproduction package should additionally include candidate hashes, incumbent and reference versions, task provenance, evaluator prompts and versions, calibration records, raw aggregate scores, exclusion rules, and canary logs. Privacy- or security-sensitive task content may be withheld, but the paper should explain the resulting reproducibility limitation. No real-agent code or dataset is claimed to accompany this version.

## 9 Discussion

## 9.1 Generation Is Not Adoption

Candidate generators optimise the production of plausible modifications. Adoption mechanisms decide whether evidence justifies changing persistent state. If the same agent proposes, evaluates, and endorses changes using overlapping evidence, a self-confirming loop can arise. EGE makes evidence separation explicit without requiring human approval for every low-impact update. Human review remains appropriate for highimpact changes or uncertain evaluators. This separation also clarifies accountability: the candidate generator is responsible for proposing a change, the evaluation service is responsible for producing a measurement under a fixed protocol, and the governance authority is responsible for defining which risks and trade-ofs are acceptable. These responsibilities may be implemented by separate services or teams, but they should remain distinguishable in logs and incident reviews.

## 9.2 Evaluator Validity

Fresh audit data do not guarantee evaluator correctness. The candidate-specific calibration construction accounts for sampling uncertainty in the observed discrepancy, but it does not repair an invalid reference construct or a calibration sample drawn from the wrong population. Candidate-specific calibration is expensive and may itself be vulnerable if the reference procedure is not blinded or independent. A model-based judge may favour a candidate’s style or exploit a scoring rubric. Deterministic verifiers and blinded human adjudication are preferable where feasible. For subjective tasks, inter-rater agreement and construct validity matter as much as nominal sample size. A numerically precise interval around a systematically invalid construct is not evidence that the intended capability has been measured.

## 9.3 Distribution Shift and Unmeasured Failure

The statistical result concerns the declared distribution $\mathcal { D } _ { r }$ . It does not automatically transfer to shifted deployment tasks. Monitoring, re-audit, and suspension of automatic adoption may be needed when representativeness deteriorates. No finite metric set can guarantee preservation of capabilities that are not measured. Distribution monitoring should therefore be treated as a trigger for reassessment, not as proof that the original guarantees remain valid. A shift detector can itself produce false alarms or miss subtle changes, so its thresholds and response policy should be documented. When a material shift is detected, the organisation may suspend automatic adoption, collect a new representative audit set, revise the target distributions, or require additional human review. Any change to the reference distribution or protected metric definitions should be versioned.

## 9.4 Conservatism and Cost

The allocation in (17) is intentionally conservative: later candidate indices receive smaller per-test error levels. This can require larger samples. Alternatives include a finite prespecified candidate horizon, confidence sequences, or more eficient concentration bounds. Any replacement must preserve the relevant family-wise guarantee under its own assumptions. There is also a distinction between the error budget and the desired power of the procedure. The alpha-spending schedule controls false statistical acceptance, but it does not guarantee that beneficial candidates will be accepted. Later candidates receive smaller error allocations, which can increase the required sample size and the probability of an inconclusive outcome at a fixed sample size. A deployment study should therefore report both error control and power or acceptance probability under specified efect sizes. The system owner must choose an acceptable balance between audit cost, decision delay, and missed improvements.

## 10 Limitations and Future Work

The guarantee assumes bounded paired observations and conditionally independent fresh confirmatory and calibration samples. Real agent tasks may be dependent, nonstationary, or afected by shared infrastructure. The reference-evaluation procedure must be suficiently valid for the intended construct; the theorem only controls discrepancy relative to that procedure. Candidate-specific calibration is sample-intensive, and the Monte Carlo study is a stylised stress test rather than a real-agent evaluation. Extensions to clustered sampling, martingale diferences, nonstationary evaluation, and eficient uniform calibration over candidate classes are important. A reference can become obsolete; the bridge-test result controls only declared metrics on a common distribution. Configuration rollback cannot undo external actions. The immediate empirica priority is to implement the protocol in a representative agent system and measure both audit cost and operational outcomes.

A further limitation concerns metric multiplicity within a capability. A single aggregate score can conceal subgroup regressions, while a large collection of subgroup metrics increases the number of simultaneous tests and therefore the required error allocation. Hierarchical testing or structured metric families may improve eficiency, but they require a prespecified hierarchy and a valid multiplicity procedure. Another limitation is that bounded normalised scores can discard information about severity. For example, a rare catastrophic event may be poorly represented by an average score in [0, 1]. Such outcomes should be represented by separate hard constraints, tail-risk measures, or event-rate requirements rather than being diluted in an average capability score.

The framework also assumes that the comparator can be evaluated on the same instance as the candidate. This paired design is attractive because it removes some task-to-task variation, but it may not be feasible when candidate changes alter task eligibility, tool availability, or execution pathways. In such cases, the experiment may require matched cohorts, randomised assignment, or a diferent estimand. The resulting inference must reflect the actual design; the paired Hoefding formula should not be applied mechanically.

Finally, EGE is a governance framework, not a complete agent security architecture. It does not by itself prevent prompt injection, data exfiltration, supply-chain compromise, malicious tool responses, or failures in the underlying model. Permission isolation, secure execution, logging, identity management, and incident response remain necessary. The contribution is narrower: EGE specifies a statistically explicit gate for persistent adoption and a deployment protocol that complements those controls.

## 11 Conclusion

This paper developed Evidence-Governed Evolution for the persistent adoption of self-generated modifications to autonomous-agent configurations. The framework separates candidate generation from confirmatory adoption, requires fresh paired evaluation, calibrates the operational evaluator against an independently governed reference procedure, protects declared capabilities relative to a versioned reference, and places accepted candidates behind bounded canary deployment. Under explicit conditional sampling assumptions, the evaluator-calibrated lower bounds provide simultaneous coverage over a countable sequence of adaptively generated candidates. A reference-lineage result makes the cost of replacing a protected baseline explicit, while an audit-planning inequality relates the efect-size margin to confirmatory and calibration sample requirements.

The Monte Carlo stress test demonstrates one narrow point: candidate-dependent evaluator optimism can make an evaluator-naive gate accept harmful updates, while candidate-specific calibration can prevent those acceptances in a stylised setting. It does not establish that EGE improves real-agent performance, is operationally eficient across realistic workloads, or remains valid under arbitrary dependence and distribution shift. Those questions require a controlled agent-task study and careful validation of the reference evaluator. EGE should therefore be viewed as a formal adoption protocol with testable assumptions, not as a proof of universal agent safety. The central principle remains that generating a plausible update is not evidence that it should become persistent system state.

## References

A. Agrawal, N. Kedia, A. Panwar, J. Mohan, N. Kwatra, B. S. Gulavani, A. Tumanov, and R. Ramjee, “Taming throughput-latency tradeof in LLM inference with Sarathi-Serve,” in Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation, pp. 117–134, 2024.

D. Amodei et al., “Concrete problems in AI safety,” arXiv:1606.06565, 2016.

A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, and T. Schuster, “Conformal risk control,” in International Conference on Learning Representations, 2024.

S. Boucheron, G. Lugosi, and P. Massart, Concentration Inequalities: A Nonasymptotic Theory of Independence. Oxford University Press, 2013.

Y. Chow, M. Ghavamzadeh, L. Janson, and M. Pavone, “Risk-constrained reinforcement learning with percentile risk criteria,” Journal of Machine Learning Research, vol. 18, no. 167, pp. 1–51, 2018.

D. L. DeMets and K. K. G. Lan, “Interim analysis: The alpha spending function approach,” Statistics in Medicine, vol. 13, no. 13–14, pp. 1341–1352, 1994, doi: 10.1002/sim.4780131308.

C. Dwork, V. Feldman, M. Hardt, T. Pitassi, O. Reingold, and A. Roth, “The reusable holdout: Preserving validity in adaptive data analysis,” Science, vol. 349, no. 6248, pp. 636–638, 2015, doi: 10.1126/science.aaa9375.

H.-A. Gao et al., “A survey of self-evolving agents: What, when, how, and where to evolve on the path to artificial super intelligence,” Transactions on Machine Learning Research, 2026.

X. Gao et al., “EvoAgentBench: Benchmarking agent self-evolution via ability transfer,” in Findings of the Association for Computational Linguistics: EMNLP 2026, pp. 50387–50402, 2026, doi: 10.18653/2026.findingsemnlp.2344.

J. García and F. Fernández, “A comprehensive survey on safe reinforcement learning,” Journal of Machine Learning Research, vol. 16, no. 42, pp. 1437–1480, 2015.

Q. Guo et al., “Connecting large language models with evolutionary algorithms yields powerful prompt optimizers,” in International Conference on Learning Representations, 2024.

D. Hendrycks et al., “Measuring massive multitask language understanding,” in International Conference on Learning Representations, 2021.

W. Hoefding, “Probability inequalities for sums of bounded random variables,” Journal of the American Statistical Association, vol. 58, no. 301, pp. 13–30, 1963, doi: 10.1080/01621459.1963.10500830.

S. R. Howard, A. Ramdas, J. McAulife, and J. Sekhon, “Time-uniform, nonparametric, nonasymptotic confidence sequences,” The Annals of Statistics, vol. 49, no. 2, pp. 1055–1080, 2021, doi: 10.1214/20- AOS1991.

W. Kwon et al., “Eficient memory management for large language model serving with PagedAttention,” in Proceedings of the 29th Symposium on Operating Systems Principles, pp. 611–626, 2023, doi: 10.1145/3600006.3613165.

K. K. G. Lan and D. L. DeMets, “Discrete sequential boundaries for clinical trials,” Biometrika, vol. 70, no. 3, pp. 659–663, 1983, doi: 10.1093/biomet/70.3.659.

R. Laroche, P. Trichelair, and R. T. des Combes, “Safe policy improvement with baseline bootstrapping,” in Proceedings of the 36th International Conference on Machine Learning, PMLR, vol. 97, pp. 3652–3661, 2019.

M. Lin et al., “Harness updating is not harness benefit: Disentangling evolution capabilities in self-evolving LLM agents,” in Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing, pp. 52829–52852, 2026, doi: 10.18653/2026.emnlp-main.2326.

A. Madaan et al., “Self-Refine: Iterative refinement with self-feedback,” in Advances in Neural Information Processing Systems, vol. 36, 2023.

A. Maurer and M. Pontil, “Empirical Bernstein bounds and sample variance penalization,” in Proceedings of the 22nd Annual Conference on Learning Theory, 2009.

E. Perez et al., “Red teaming language models with language models,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 3419–3448, 2022.

Z. Ren et al., “Self-improvements in modern agentic systems: A survey,” arXiv:2607.13104, 2026.

N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao, “Reflexion: Language agents with verbal reinforcement learning,” arXiv:2303.11366, 2023.

G. Wang et al., “Voyager: An open-ended embodied agent with large language models,” arXiv:2305.16291, 2023.

Z. Zhang, X. Jiang, J. Gao, Z. Yang, Y. Huang, J. Zhao, X. Chu, and Y. Wang, “Endogenous observations in agent self-evolution: A survey,” in Findings of the Association for Computational Linguistics: EMNLP 2026, pp. 25257–25276, 2026, doi: 10.18653/2026.findings-emnlp.1196.

## A Hoefding Bound Derivation

For independent $X _ { i } \in [ - 1 , 1 ]$ with common mean $\mu ,$ Hoefding’s inequality gives

$$
\mathbb { P } ( \bar { X } - \mu \geq u ) \leq \exp \left( - \frac { 2 n u ^ { 2 } } { ( 1 - ( - 1 ) ) ^ { 2 } } \right) = \exp ( - n u ^ { 2 } / 2 ) .
$$

Setting this upper bound equal to a yields $u = \sqrt { 2 \ln ( 1 / a ) / n }$ , and hence

$$
\mathbb { P } \left\{ \mu \geq \bar { X } - \sqrt { \frac { 2 \ln ( 1 / a ) } { n } } \right\} \geq 1 - a .
$$

This is the evaluation radius used in (14); variance-sensitive alternatives such as empirical Bernstein bounds can be tighter when their assumptions are appropriate (Maurer and Pontil, 2009; Boucheron et al., 2013).

## B Calibration-Bound Derivation

For independent calibration observations $W _ { i } \in [ - 2 , 2 ]$ with mean $b ,$ the one-sided Hoefding inequality gives

$$
\operatorname* { P r } ( b - { \bar { W } } \geq u ) \leq \exp \left( - { \frac { 2 m u ^ { 2 } } { ( 2 - ( - 2 ) ) ^ { 2 } } } \right) = \exp ( - m u ^ { 2 } / 8 ) .
$$

Setting the right-hand side to c yields $u = \sqrt { 8 \ln ( 1 / c ) / m }$ . Therefore $\operatorname* { P r } ( b \leq \bar { W } + u ) \geq 1 - c ,$ which establishes the one-sided calibration upper bound in (15).

## C Evaluation and Calibration Sample-Size Bounds

To make the evaluation radius no greater than $h _ { E } > 0$ , it is suficient that

$$
n _ { j , r } \geq \frac { 2 \ln ( 1 / e _ { j , r } ) } { h _ { E } ^ { 2 } } .\tag{24}
$$

To make the calibration radius no greater than $h _ { b } > 0$ , it is suficient that

$$
m _ { j , r } \geq \frac { 8 \ln ( 1 / c _ { j , r } ) } { h _ { b } ^ { 2 } } .
$$

For a target-efect gap g, the suficient condition $2 h _ { E } + 2 h _ { b } \leq g$ yields the acceptance-probability bound described in Section 5.4. These are conservative worst-case bounds; variance-adaptive alternatives require their own validity arguments.

## D Notation and Implementation Checklist

For clarity, Table 5 summarises the principal symbols used in the framework. The table is intended as an implementation aid; the formal definitions in the main text take precedence if a symbol is used in a specialised context.

A reference implementation should verify the following before enabling automatic commitment:

1. The candidate and comparator identifiers are immutable for the duration of the audit.

Table 5: Principal notation used in EGE
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td></td><td>Incumbent configuration before candidate j</td></tr><tr><td> $\underset { ^ { \ . } } { ^ { I _ { j } } }$ </td><td>Frozen candidate configuration</td></tr><tr><td> $A _ { \mathrm { r e f } }$ </td><td>Versioned stable reference configuration</td></tr><tr><td> $\mathcal { E } _ { j }$ </td><td>Experience available to candidate generation</td></tr><tr><td> $\check { \mathcal { I } }$ </td><td>Prespecified primary-objective test set</td></tr><tr><td> $\kappa$ </td><td>Prespecified protected-capability test set</td></tr><tr><td> $\mathcal { G }$ </td><td>Mandatory governance checks</td></tr><tr><td> $\mathcal { D } _ { r }$ </td><td>Evaluation distribution for metric r</td></tr><tr><td> $\Delta _ { i { \mathrm { ~ } } n } ^ { R }$  J,r</td><td>Reference-procedure population difference for candidate j and metric r</td></tr><tr><td> $b _ { j , r }$ </td><td>Mean operational-to-reference evaluator discrepancy</td></tr><tr><td> ${ \bar { U } } _ { \cdot } ^ { b }$   $U _ { j , r } ^ { \upsilon }$ </td><td>One-sided calibration upper confidence bound on  $b _ { j , r }$ </td></tr><tr><td> $e _ { j , r } , c _ { j , r }$ </td><td>Evaluation and calibration error allocations</td></tr><tr><td> $m _ { j , r }$ </td><td>Number of calibration observations</td></tr><tr><td> $\delta _ { r }$ </td><td>Minimum meaningful primary improvement</td></tr><tr><td> $\epsilon _ { r }$ </td><td>Maximum tolerated protected-capability degradation</td></tr><tr><td> $_ \alpha$ </td><td>Global family-wise error budget</td></tr><tr><td> ${ n _ { j , r } }$ </td><td>Number of confirmatory paired observations</td></tr><tr><td> $L _ { j , r } ^ { R }$ </td><td>Evaluator-calibrated lower confidence bound for the reference estimand</td></tr></table>

2. The candidate index is allocated by a persistent audit controller and cannot be reset after rejection.

3. The primary and protected metric families, their comparators, thresholds, and evaluation distributions are versioned.

4. The audit sample is fresh for the frozen candidate and is not fed back into its development loop.

5. The sampling unit and dependence assumptions match the task-generation process.

6. The score range is enforced or justified; observations outside the assumed interval are treated as protocol failures.

7. The candidate-specific calibration sample is independent of the confirmatory sample and is drawn from the declared evaluation distribution.

8. Every hard governance check has a deterministic or authorised review procedure and a defined failure action.

9. Canary exposure is bounded, monitored, and stopped by a controller independent of the candidate.

10. Rollback, state restoration, and compensating actions are tested before deployment.

This checklist does not add new statistical guarantees. It translates the assumptions and protocol requirements into auditable implementation conditions.

## E Worked Illustrative Decision

Consider one primary test and two protected tests for candidate $j = 1$ , with $R = 3$ and global error budget $\alpha = 0 . 0 5$ . Under (17), each evaluation and calibration allocation is $3 \alpha / ( \pi ^ { 2 } R )$ . For each metric, the controller collects independent confirmatory and calibration samples, computes $L _ { j , r } ^ { E }$ from (14), computes $U _ { j , \prime } ^ { b }$ from (15), and sets $L _ { j , r } ^ { R } = L _ { j , r } ^ { E } - U _ { j , r } ^ { b }$ . The candidate passes only if the primary lower bound exceeds its prespecified meaningful-improvement margin and both protected lower bounds exceed their respective non-inferiority thresholds. This example is procedural: the margins must be selected from application requirements, not tuned after observing results.

A positive observed operational-evaluator mean is insuficient. If the evaluator discrepancy is large or poorly calibrated, the candidate may remain inconclusive even when the operational score is high. Likewise, failure of one protected test blocks acceptance even if the primary objective improves substantially. Governance checks are evaluated separately and cannot be ofset by a numerical score.