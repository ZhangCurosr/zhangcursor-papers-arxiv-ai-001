# Beyond Answers, Traces, and Eficiency: Evidence for Adaptive Reasoning

Sejin Kim<sup>1</sup>, Sundong Kim<sup>2</sup>,

<sup>1</sup>Chungnam National University <sup>2</sup>Gwangju Institute of Science and Technology sejin.kim@cnu.ac.kr, sundong@gist.ac.kr

## Abstract

Current AI systems are evaluated as reasoners through finalanswer accuracy, reasoning traces, process supervision, and computational or human-relative eficiency. Yet these measures do not establish whether reasoning adapts when tasks or evidence change. We distinguish across-task adaptivity, where acquired abstractions transfer or recombine across novel problems, from within-task adaptivity, where hypotheses and actions are revised in response to new observations, feedback, or counterexamples. Using ARC-AGI as a diagnostic case study rather than an alignment benchmark, we examine what existing measures leave unresolved when testtime search, sampling, or adaptation afects performance. We propose intervention-oriented evaluation that tests abstraction transfer and hypothesis revision under controlled changes to tasks and evidence, with explicit reporting of test-time procedures and budgets. This perspective supports evaluating systems for robustness, responsiveness to corrective feedback, and auditability under changing conditions.

## Introduction

Reasoning evaluation has traditionally focused on outcomebased measures such as final-answer accuracy and pass rates (Uesato et al. 2022; Lee et al. 2025). As reasoning systems have become more complex, evaluation has expanded to intermediate reasoning traces, process-level supervision, inference-time computation, and human-relative eficiency. Chain-of-thought (CoT), process reward models (PRMs), and metrics such as relative human action eficiency (RHAE) exemplify this shift beyond answer correctness (Wei et al. 2022; Lightman et al. 2024; ARC Foundation 2026). Together, these developments reflect a growing recognition that final outcomes provide limited evidence about how a solution was obtained.

Yet an important question remains: does the system change its reasoning appropriately when the task or available evidence changes? A correct answer does not establish that an acquired abstraction will transfer to a novel problem. A plausible reasoning trace does not establish that the reasoning will change when new evidence contradicts an earlier hypothesis. Likewise, eficient behavior does not reveal what the system learned or whether it can reuse that knowledge. Outcome quality, trace quality, and eficiency therefore provide useful but incomplete evidence of adaptive reasoning.

This gap becomes more consequential when systems rely on test-time search, sampling, refinement, or interaction (Wang et al. 2023; Snell et al. 2025; Jeon et al. 2026). Such procedures can be legitimate components of reasoning, but higher performance alone does not reveal whether improvement comes from abstraction transfer, hypothesis revision, or simply more opportunities for selection. This makes explicit test-time reporting and stronger behavioral evidence important for interpreting reasoning claims.

Recent work has framed LLM reasoning in terms of System-2 thinking (Zhang et al. 2026; Kim and Kim 2024). We argue that adaptive reasoning requires evidence at two complementary scales. Across-task adaptivity concerns whether acquired abstractions can be transferred, reused, or recombined across novel problems. Within-task adaptivity concerns whether hypotheses and decisions are revised as observations, feedback, or counterexamples become available. This distinction shifts evaluation from observing reasoning to testing how behavior responds to controlled changes in tasks and evidence. Stronger evidence for adaptive reasoning therefore comes from behavior under intervention, rather than from a successful trajectory alone.

Adaptive reasoning is not equivalent to alignment, but both adaptive capabilities matter for systems operating under novelty, ambiguity, and human feedback. Such systems must often reuse relevant knowledge while revising assumptions that no longer fit the available evidence. Evidence of transfer and revision can therefore inform alignment-relevant properties such as robustness, responsiveness to corrective feedback, and auditability.

This paper makes three contributions. First, we organize existing reasoning evaluations by the evidence they provide about outcomes, reasoning traces and processes, resource use, and human-relative eficiency. Second, we characterize adaptive reasoning through across-task transfer and withintask revision, and argue for evaluating both through controlled changes to tasks and evidence. Third, using ARC-AGI as a diagnostic case study, we outline an intervention-oriented evaluation perspective that explicitly reports test-time procedures and computational budgets.

## What Existing Reasoning Evaluations Tell Us

Reasoning evaluations provide diferent kinds of evidence about model behavior. Our aim is not to rank or replace existing metrics, but to clarify what each establishes and what it leaves unresolved. These forms of evidence address related but diferent questions: whether a task was solved, what reasoning was expressed, whether intermediate steps were valid, what resources were required, and how eficiently behavior compared with humans. We view them as complementary forms, each supporting a diferent class of reasoning claims rather than progressively suficient definitions of reasoning.

Outcome-based evidence. Accuracy, Pass@k, and success rate directly measure whether a system solves a task and remain fundamental indicators of task competence (Uesato et al. 2022; Lee et al. 2025). Their limitation is not that they are uninformative, but that the same successful outcome can arise from substantially diferent procedures (Hwang et al. 2025). A correct solution may reflect a transferable abstraction, a memorized pattern, a brittle heuristic, extensive search, or large-scale candidate sampling. This outcome equivalence makes final performance insuficient for distinguishing how a solution was obtained or what knowledge would persist beyond the evaluated instance. Outcome metrics therefore establish task success, but do not by themselves establish the reasoning process responsible for it.

Trace-based evidence. Reasoning traces such as chainof-thought, natural-language rationales, scratchpads, or intermediate programs expose information that final answers omit (Wei et al. 2022; Lee et al. 2025). They can reveal candidate hypotheses, decompositions, and intermediate representations, making a solution more inspectable. However, an observed trace is an artifact of reasoning, not by itself evidence that the underlying behavior is adaptive. A rationale may be weakly coupled to the decision that produced the answer (Turpin et al. 2023), and observing a successful trace does not establish how the system would respond if relevant evidence changed. Moreover, even a faithful trace answers only part of the question. Faithfulness concerns whether the trace reflects the process associated with a decision, whereas adaptivity concerns whether that process changes appropriately under diferent evidence.

Process-based evidence. Process-level evaluation goes further by assessing intermediate reasoning steps rather than merely exposing them. Process reward models (PRMs), for example, can evaluate whether individual steps are valid or useful and provide denser supervision than outcome-only rewards (Uesato et al. 2022; Lightman et al. 2024). This is an important advance, but step quality and evidence responsiveness are distinct evaluation axes. A trajectory may consist of locally valid steps without establishing whether the system would revise an earlier hypothesis when subsequent observations contradict it. A high process score can therefore support a claim about the quality of an observed trajectory without establishing its behavior under changed evidence. Process evaluation asks whether intermediate reasoning is valid; adaptive evaluation additionally asks whether that reasoning updates appropriately when its evidential basis changes.

<table><tr><td>Evidence</td><td>What it establishes / leaves unresolved</td></tr><tr><td>Outcome</td><td>Measures task success; does not identify how the solution was obtained.</td></tr><tr><td>Trace</td><td>Exposes expressed reasoning; does not es- tablish behavioral grounding or adaptation.</td></tr><tr><td>Process</td><td>Evaluates intermediate-step quality; does not test responsiveness to changed evidence.</td></tr><tr><td>Resource</td><td>Reports compute and search budgets; does not separate stronger reasoning from larger</td></tr><tr><td>Human-relative</td><td>search. Measures efficiency against humans; does not identify what was acquired or revised.</td></tr></table>

Table 1: Complementary evidence provided by existing reasoning evaluations and the questions each leaves unresolved.

Resource-based evidence. The growing use of test-time search, sampling, refinement, and additional inference compute introduces another dimension to reasoning evaluation (Wang et al. 2023; Brown et al. 2024; Snell et al. 2025; Jeon et al. 2026). Systems achieving similar accuracy may require substantially diferent numbers of samples, tokens, search steps, refinement rounds, or interactions. Reporting these quantities is important because it separates achieved performance from the resources used to obtain it. Additional computation is not inherently undesirable: search, iteration, and refinement can themselves be components of reasoning (Pourcel, Colas, and Oudeyer 2025). The unresolved issue is interpretive. An accuracy gain may result from a better abstraction or more efective hypothesis revision, but it may also result from a larger opportunity to generate, search, and select candidates. Resource reporting makes this distinction more visible without resolving it on its own.

Human-relative evidence. Human-centric metrics add a behavioral reference point by comparing how eficiently humans and AI systems solve the same task. Relative human action eficiency (RHAE), used in ARC-AGI-3, captures differences that final accuracy cannot: a human and a model may both succeed while requiring substantially diferent numbers of actions (ARC Foundation 2026). Such comparisons provide useful evidence about behavioral eficiency and can expose gaps hidden by binary success measures. Human trajectories can provide still richer references, including common action sequences, strategy distributions, and points at which behavior changes (Kim et al. 2026; LeGris et al. 2025). However, human-like eficiency should not be treated as a normative definition of good reasoning, nor does trajectory similarity establish that humans and models acquired the same abstractions or revised the same hypotheses (Kim, Lee, and Kim 2025). Human-relative evidence therefore characterizes behavioral eficiency and trajectory patterns, while leaving the underlying adaptive capability unresolved.

Tab. 1 summarizes these complementary forms. Each expands the evidence available for reasoning claims, but none directly tests whether reasoning itself changes appropriately when relevant task properties or available evidence change. This remaining question motivates our account of adaptive reasoning in the next section.

![](images/397c8c85e1654b677d2ad487e171912c470d854296fe885909444ffeaaf8899b.jpg)  
Figure 1: Existing evidence characterizes what was observed; intervention-oriented evaluation tests how behavior changes under controlled changes to tasks or evidence.

## What Is Missing? Evidence for Adaptivity

The evidence forms above characterize behavior on a particular task or trajectory, but claims of adaptive reasoning require evidence that controlled changes in relevant task conditions or available evidence induce systematic, corresponding behavioral changes. We treat adaptivity as a behavioral property rather than a claim about a particular internal mechanism. We distinguish two complementary scales: across-task adaptivity concerns whether acquired abstractions transfer or recombine across changing tasks, whereas within-task adaptivity concerns whether hypotheses and subsequent decisions are revised as task-specific evidence changes.

Behavioral notation. Let τ denote a task, $E _ { t }$ the evidence available by step t, with $E _ { 0 }$ denoting the initial task evidence, and q a fixed test-time budget. We write $B _ { \theta } ( \tau , E _ { t } ; q )$ for the observable behavior of a system θ under these conditions. Let $C ( \tau )$ denote the evaluator-specified task structure relevant to the reasoning claim. It specifies what a controlled task variation is intended to preserve or change; it need not correspond to an accessible internal representation of the model.

Across-task adaptivity: transferring and recombining abstractions. For across-task adaptivity, consider a controlled intervention $I _ { T }$ producing $\bar { \tau ^ { \prime } } = \bar { I _ { T } } ( \tau )$ . We define

$$
\Delta { C } = C ( \tau ^ { \prime } ) \Theta C ( \tau ) , \Delta { B } = B _ { \theta } ( \tau ^ { \prime } , E _ { 0 } ; q ) \Theta B _ { \theta } ( \tau , E _ { 0 } ; q ) ,
$$

where $\ominus$ denotes an evaluator-specified, task-appropriate contrast between structures or behaviors. The central evidential relation is

$$
\begin{array} { r } { \boxed { \Delta C \sim \Delta B } , } \end{array}
$$

where ∼ denotes an evaluator-defined consistency relation between the task change and the behavioral response. It is not a universal similarity metric, but specifies what behavior should remain stable and what should change under the intervention.

This formulation makes across-task adaptivity more specific than generic generalization. Standard generalization asks whether a model performs well on unseen examples. $\mathbf { A }$ model can satisfy this criterion while relying on correlations unrelated to the abstraction implied by a reasoning claim. Across-task adaptivity instead asks whether behavior tracks the structure deliberately preserved, recombined, or altered across related tasks.

A useful distinction is between concept-preserving and concept-changing interventions. In a concept-preserving intervention, surface properties such as object identities, layouts, ordering, or irrelevant context may change while the relation required for the solution remains fixed. Here, $C ( \tau ^ { \prime } ) =$

$C ( \tau )$ with respect to the target abstraction, so the corresponding behavior should remain stable or transform predictably. In a concept-changing intervention, the relevant structure itself changes, and adaptive behavior should change consistently with that modification. Evidence for transfer therefore combines stability to changes that should not matter with sensitivity to changes that should.

This criterion does not require identical reasoning traces or action sequences across tasks. The same abstraction may support diferent strategies in diferent contexts. Nor does it require accuracy to remain constant under every variation. A harder recombination may reduce performance while still revealing reuse of the relevant structure. The evidential target is the dependency between controlled structural variation and corresponding behavior, rather than performance invariance alone.

Within-task adaptivity: revising hypotheses as evidence changes. Early observations may support several interpretations, while later counterexamples, feedback, or interaction outcomes may invalidate an intermediate hypothesis. Prior work has explored iterative refinement and hypothesis revision using such signals (Madaan et al. 2023; Shinn et al. 2023; Wang et al. 2024). Within-task adaptivity asks whether subsequent reasoning changes specifically in response to the evidence rather than merely persisting with an initial interpretation.

Let $I _ { E }$ denote an evidence intervention at step t, producing

$$
E _ { t } ^ { \prime } = I _ { E } ( E _ { t } ) , \qquad \Delta E = E _ { t } ^ { \prime } \ominus E _ { t } .
$$

Because an evaluator may not observe an explicit internal hypothesis, let $Z _ { \theta } ( \tau , E _ { t } ; \dot { q } )$ denote an observable intermediate decision, such as a hypothesis, proposal, program, action plan, or tool call. The relevant dependency is

$$
\begin{array} { r } { \boxed { \Delta E \longrightarrow \Delta Z _ { t } \longrightarrow \Delta B _ { > t } } , } \end{array}
$$

where $\Delta Z _ { t }$ denotes the corresponding change in the observable intermediate decision and $\Delta B _ { > t }$ the subsequent behavioral change. The revision should be systematically consistent with what the new evidence supports or rules out.

Behavioral change alone is insuficient. Diferent outputs can result from stochastic sampling, repeated retries, or unrelated variation. If new evidence contradicts one component of a hypothesis, stronger evidence of revision is obtained when the corresponding component of $Z _ { t }$ changes rather than when the system changes arbitrarily. Conversely, evidence consistent with the current hypothesis should not induce unrelated revision. The evaluation need only establish a meaningful relationship between changes in evidence and changes in reasoning, without prescribing a unique update.

This distinction is particularly important for separating revision from repeated search. Two systems may initially fail and later produce the same correct answer. One may incorporate information about why an earlier proposal failed and modify the relevant part of its next proposal. Another may generate additional independent candidates until one succeeds. Both are legitimate inference strategies, but they support diferent claims. The former provides direct behavioral evidence of evidence-sensitive revision, whereas the latter primarily demonstrates the efectiveness of additional sampling or selection. Holding the test-time budget q explicit helps prevent these diferences from being obscured by unequal search opportunities.

Why are both scales needed? Across-task and withintask adaptivity capture related but non-interchangeable properties. A system may transfer a useful abstraction across problems yet fail to revise it when contradictory evidence appears within a task. Conversely, a system may respond efectively to local feedback while relying on representations that do not transfer beyond the current instance. Together, the two scales examine whether reasoning can both reuse knowledge across changing contexts and revise it when the available evidence no longer supports it.

What counts as adaptive evidence? Across both scales, three properties strengthen an adaptivity claim. First, relevance: behavior should respond to changes that matter to the current abstraction or hypothesis while remaining relatively stable to irrelevant variation. Second, specificity: the component of reasoning that changes should correspond to the information altered by the intervention rather than reflecting arbitrary global change. Third, consistency: comparable interventions should produce comparable response patterns across instances rather than isolated stochastic changes. These criteria are not intended as a universal scalar metric; they specify the behavioral relationship that an evaluation should expose.

Observation is not intervention. Outcome, trace, process, resource, and human-relative evaluations primarily characterize behavior that has already occurred. Controlled variants have also been used to probe model behavior under targeted changes in prior evaluation work (Gardner et al. 2020; Ribeiro et al. 2020). Here, intervention-oriented evidence changes a variable that the claimed reasoning should depend on and measures the corresponding response. Across tasks, this can mean preserving or modifying a relevant structure; within a task, it can mean adding an observation, counterexample, constraint, or feedback signal that changes support for a current hypothesis. The result is a paired comparison between a change in relevant information and a subsequent behavioral response.

Instead of asking only what reasoning was produced, adaptive evaluation asks how reasoning changes when relevant evidence changes. Fig. 1 summarizes how this perspective extends existing forms of reasoning evidence.

These forms of evidence remain complementary, while controlled intervention adds distinct evidence that directly probes the transfer and revision dependencies defined above.

Behavioral interventions do not identify a model’s complete internal causal mechanism. Diferent mechanisms may produce similar responses, and determining whether an intervention cleanly preserves or alters only the intended structure can itself be dificult. Our claim is narrower. Controlled behavioral variation provides stronger evidence for abstraction transfer and hypothesis revision than observing a successful trajectory in isolation. We treat across-task and within-task adaptivity as a minimal diagnostic framework for adaptive reasoning claims, not as a complete definition of reasoning.

## ARC-AGI as a Diagnostic Case Study

ARC-AGI provides a useful diagnostic case because it spans two related but distinct reasoning settings. ARC-AGI-1 and ARC-AGI-2 use static input–output tasks, where a solver infers a transformation from sparse demonstrations and applies it to a novel input (Chollet 2019; Chollet et al. 2025). ARC-AGI-3 instead introduces interactive environments in which an agent acts, observes consequences, and reasons from subsequent evidence (ARC Foundation 2026). Together, we use them as complementary diagnostics, not alignment benchmarks, for probing abstraction transfer across controlled task variations and evidence-sensitive revision during interaction.

Across-task adaptivity in ARC-AGI-1/2. In ARC-AGI-1 and ARC-AGI-2, evaluation ultimately emphasizes whether the correct output is produced (Chollet 2019; Chollet et al. 2025). Yet identical outcomes may arise from diferent testtime procedures: a solver may infer a compact transformation, search candidate programs, sample and rank hypotheses, refine an earlier proposal, or adapt at test time (Jeon et al. 2026; Pourcel, Colas, and Oudeyer 2025; Akyürek et al. 2025). Final accuracy does not reveal which capability accounts for success, while resource reporting alone does not establish whether an inferred abstraction transfers when a task changes.

ARC tasks also support controlled variation. Surface properties such as colors, object identities, positions, orientations, or distractors can change while preserving a relevant relation, whereas the relation itself can change while other properties remain similar (Moskvichev, Odouard, and Mitchell 2023). Such comparisons test whether behavior tracks the structure preserved or changed by an intervention rather than merely succeeding independently on related tasks.

Within-task adaptivity in ARC-AGI-3. ARC-AGI-3 shifts from static transformation inference to sequential interaction (ARC Foundation 2026). An agent acts, observes the consequence, and obtains evidence that may support or contradict its current interpretation. Eventual success, however, may arise from evidence-sensitive revision or simply continued exploration and search. The relevant distinction is whether subsequent decisions change specifically in response to what earlier interactions revealed.

Relative human action eficiency (RHAE) adds an eficiency perspective by comparing the actions humans and agents require to succeed (ARC Foundation 2026). Human trajectories can further reveal strategies, failures, and points of revision (Kim et al. 2026; LeGris et al. 2025). Neither eficient behavior nor trajectory similarity establishes that the same abstraction was acquired or that a hypothesis was revised for the same reason (Kim, Lee, and Kim 2025). They are diagnostic references rather than normative definitions of reasoning.

Fig. 2 summarizes how the two scales of adaptivity introduced in Sec. 3 can be instantiated in ARC-AGI. Across tasks, controlled variations probe abstraction transfer; within a task, interaction probes evidence-sensitive revision. The next section turns these diagnostic distinctions into concrete intervention-oriented evaluation protocols.

![](images/7f599a2a6d09058f6b7e4e4108dacae31dd7c44d25a6fd75ace3cf78dacd41ab.jpg)  
Figure 2: ARC-AGI as a diagnostic setting for adaptive reasoning. (a) Across-task adaptivity can be probed by controlled task variations that preserve or modify the relevant structure, testing whether behavioral change tracks structural change. (b) Within-task adaptivity can be probed during interaction by asking whether new evidence induces corresponding revisions in intermediate decisions and subsequent behavior, rather than merely triggering additional search.

## Toward Intervention-Oriented Evaluation

The preceding analysis suggests evaluating adaptive reasoning through paired behavioral comparisons rather than successful trajectories in isolation. The aim is to make claims about transfer and revision testable under controlled changes without prescribing a single benchmark or scalar score. An intervention-oriented evaluation should specify what is changed, what remains fixed, what behavioral response is expected, and what test-time resources are available.

Across-task protocol. Across-task adaptivity can be evaluated by constructing paired task variants from a base task τ. Each intervention should identify the task structure C(τ) relevant to the reasoning claim and distinguish changes intended to preserve that structure from those intended to modify it. Concept-preserving variants may alter surface properties, object identities, layouts, or other irrelevant context while keeping the target relation fixed. Concept-changing variants instead modify the relation that should determine the solution while preserving unrelated structure as much as possible.

The system is then evaluated on paired tasks under the same test-time budget q. Adaptivity does not require identical traces or equal accuracy across variants. Rather, behavior should remain stable or transform predictably when the relevant structure is preserved, and change appropriately when that structure is modified. Both intervention types are important: invariance alone can arise when a system ignores the manipulated variable, while sensitivity alone can reflect general instability. Stronger evidence therefore combines robustness to irrelevant variation with responsiveness to relevant structural change. Use multiple matched variants when possible to reduce dependence on idiosyncratic examples.

Intervention construction itself afects evaluation validity. A nominally concept-preserving change may alter task difficulty or introduce a new cue, while a concept-changing intervention may modify several relevant properties at once. Variants should therefore be as localized as possible, with preserved and changed properties stated explicitly (Gardner et al. 2020; Ribeiro et al. 2020). Matched controls can test whether the same manipulation afects behavior when the altered feature is irrelevant to the target relation. Reporting the intervention procedure alongside behavior helps distinguish intervention failures from failures of the reasoning system.

Within-task protocol. Within-task adaptivity requires a paired comparison around a point at which new evidence becomes available. At step t, the evaluator records an observable intermediate decision $Z _ { t } ,$ such as a hypothesis, proposal, program, plan, or action. An intervention then changes E<sub>t</sub> to $\mathrm { \bar { \boldsymbol { E } } } _ { t } ^ { \prime }$ through an additional observation, counterexample, failed prediction, interaction outcome, verifier signal, or corrective feedback, after which the evaluator records subsequent decisions and behavior. The expected implication of the new evidence for the current decision should be clear, enabling a targeted test of whether the corresponding reasoning component changes appropriately.

<table><tr><td>Evidence</td><td>Minimum reporting</td></tr><tr><td>Outcome</td><td>Task success pre- and post-intervention. Relevant trace, hypothesis, proposal, pro-</td></tr><tr><td>Reasoning artifact</td><td>gram, action, or other observable inter- mediate decision.</td></tr><tr><td>Test-time budget</td><td>Samples, tokens, search steps, refine- ment rounds, interactions, tools, or other inference-time resources.</td></tr><tr><td>Across-task</td><td>What structure is preserved or changed, the paired task variants, and the corre- sponding behavioral response.</td></tr><tr><td>Within-task</td><td>What evidence is added or modified, the decision before and after the intervention,</td></tr><tr><td>Human reference</td><td>and subsequent behavior. Action efficiency, trajectory, or strategy information when used as a diagnostic comparison.</td></tr></table>

Table 2: Minimum reporting for interpreting interventionoriented evidence of adaptive reasoning.

The critical question is whether the resulting change is tied to the introduced evidence. Contradictory evidence should induce revision of the implicated component when warranted, whereas irrelevant or confirming evidence should not induce arbitrary changes. Such controls distinguish evidencesensitive revision from generic instability. The comparison should also separate revision from repeated search: additional independent candidates may improve success, but provide diferent evidence from using a failure to revise the next proposal. Holding the test-time budget fixed, or explicitly reporting diferences, is therefore necessary for interpretation.

Whenever possible, comparisons should branch from the same pre-intervention history. One branch receives informative evidence, while matched branches receive no additional evidence, confirming evidence, or an irrelevant perturbation. Diferences in subsequent decisions provide a cleaner test of whether revision is attributable to the intervention rather than stochastic generation or accumulated trajectory diferences. For stochastic systems, repeated runs across random seeds can help distinguish systematic evidence-sensitive updates from isolated changes (Atıl et al. 2025). These responses need not be collapsed into a single score; instead, report them by intervention type and expected behavioral change.

Reporting adaptive evidence. Report intervention results alongside conventional outcome and process measures rather than replacing them. Tab. 2 summarizes the minimum information needed to interpret adaptive reasoning claims.

These protocols do not determine whether a model possesses a particular internal mechanism. They instead strengthen evidence for narrower claims: that an abstraction is reused across controlled task changes, or that a decision is revised in a manner linked to newly available evidence. This distinction matters for alignment-relevant reasoning systems, where robustness under novelty and responsiveness to corrective information matter beyond success on a fixed evaluation instance. The value of such evidence is therefore not that it certifies alignment, but that it makes specific claims about adaptive behavior more auditable and falsifiable.

## Relation to Alignment-Relevant Properties

Adaptive reasoning is not equivalent to alignment. A system may transfer abstractions and revise hypotheses efectively while still pursuing undesirable objectives or responding poorly to human values. Our framework is therefore narrower: across-task transfer and within-task revision provide behavioral evidence for capabilities that matter under novelty, uncertainty, and corrective feedback.

Robustness under novelty. Across-task adaptivity matters for robustness because deployment conditions rarely reproduce the exact surface properties seen during training or evaluation. A robust reasoner should preserve behavior when irrelevant properties change while remaining sensitive to changes in task structure that should afect its decision. Concept-preserving and concept-changing interventions make this distinction explicit. They provide more targeted evidence than aggregate out-of-distribution performance alone about whether a system’s behavior tracks the structure on which its reasoning claim depends.

Corrigibility and responsiveness to feedback. Withintask adaptivity is related to a system’s ability to respond to corrective information. When new observations, failures, or human feedback undermine an earlier hypothesis, an adaptive system should revise the implicated decision rather than persist or merely search for another successful output. Corrigibility is broader, encompassing acceptance of human oversight or correction of a system’s behavior or objectives (Soares et al. 2015; Hadfield-Menell et al. 2017). Our test instead asks whether corrective evidence can systematically alter subsequent reasoning in practice.

Responsiveness alone is insuficient. A system that changes its behavior indiscriminately in response to irrelevant, noisy, or adversarial feedback may appear adaptive while remaining unreliable. Alignment-relevant responsiveness depends not only on whether behavior changes, but on whether the change is appropriately tied to the evidence that should matter. The relevance and specificity criteria introduced in Sec. 3 help distinguish targeted revision from generic sensitivity.

Auditability. Intervention-oriented evaluation also makes reasoning claims easier to audit. Outcome accuracy shows whether a system succeeded, while traces and process scores reveal aspects of how it reasoned. Controlled interventions add a paired comparison between what information changed and how behavior responded. Combined with explicit reporting of test-time budgets and intermediate decisions, such comparisons make claims about transfer and revision more falsifiable and easier to distinguish from gains due primarily to additional search, sampling, or selection.

These diagnostics should complement, not replace, evaluations of objectives, preferences, policy compliance, or other alignment targets. This distinction separates adaptive competence from the desirability of the objective being pursued. Evidence of transfer or revision therefore does not certify that a system is aligned, safe, or causally understood; it provides more specific evidence for alignment-relevant claims about robustness, responsiveness to correction, and auditability under changing conditions.

## Discussion

Adaptive evidence complements existing evaluation. Our proposal does not imply that final-answer accuracy, reasoning traces, process evaluation, or eficiency measures should be displaced. Outcome metrics remain necessary for establishing whether a system can solve a task, while traces and process-level assessments expose aspects of the reasoning associated with a solution. Resource and human-relative measures further reveal computational or behavioral costs that accuracy alone obscures. The limitation is one of evidential scope: none of these measures by itself establishes whether reasoning transfers across controlled task changes or revises appropriately when relevant evidence changes. Adaptive evidence should therefore complement, rather than replace, existing evaluation.

Behavioral proxies are imperfect. The framework does not require a particular reasoning representation. A system may express intermediate hypotheses through natural language, programs, plans, actions, tool calls, or other observable decisions, while some systems may expose no explicit symbolic hypothesis at all. The variable Z is therefore a system-dependent behavioral proxy rather than a claim that all reasoners share a common intermediate representation. This flexibility broadens the framework’s applicability, but also introduces measurement uncertainty. An observable proposal or action may only partially reflect the decision process governing subsequent behavior. Claims about revision should therefore be calibrated to what the chosen observable can support.

Behavioral intervention does not identify mechanism. Controlled variation strengthens evidence about dependencies between information and behavior, but it does not uniquely identify the mechanism producing them. Diferent mechanisms may exhibit similar patterns of transfer or revision, and even localized interventions may alter task difficulty, introduce unintended cues, or afect several properties at once. Within-task interventions may also change the trajectory in ways unrelated to the intended evidence. Localized interventions, matched controls, repeated comparisons, and fixed or transparently reported test-time budgets reduce these ambiguities but cannot eliminate them. Our framework therefore supports stronger behavioral claims about transfer and revision without establishing mechanistic causality.

Adaptation must be selective. Adaptivity should not be equated with behavioral change itself. A system that changes its decisions in response to irrelevant, noisy, or adversarial information may appear responsive while remaining unreliable. Conversely, persistence can be appropriate when new information does not bear on the current hypothesis. The target is selective adaptation: stability to changes that should not matter and sensitivity to changes that should. This is why relevance, specificity, and consistency are central to interpreting both across-task and within-task evidence. This selectivity also suggests asymmetric controls. Negative controls should alter information irrelevant to the current hypothesis and test for stability, whereas positive controls should alter evidence that warrants revision and test for sensitivity. Evaluating both responses helps distinguish genuine evidence dependence from generic conservatism or instability.

Adaptive evidence should not be collapsed prematurely. Across-task transfer and within-task revision capture diferent failure modes; performance on one does not imply performance on the other. Concept-preserving and conceptchanging interventions test complementary properties: a system may appear robust because it ignores relevant changes, or appear responsive because it changes indiscriminately. An aggregate adaptivity score could obscure these distinctions. Reporting results separately by adaptivity scale, intervention type, and expected behavioral response preserves information about where a system succeeds or fails. Aggregation may still be useful for comparison, but it should follow, not replace, this disaggregated evidence.

The framework is not specific to ARC-AGI. ARC-AGI serves as a diagnostic case because its tasks make controlled variation, sparse evidence, and interactive feedback comparatively explicit. The same evaluation logic can apply whenever an evaluator can specify a task property to preserve or change, introduce evidence with a known implication, and observe a subsequent decision or action. Examples include program synthesis, planning, tool-using agents, and interactive problem solving. Concrete interventions and behavioral proxies will difer across domains, but the underlying question remains the same: does behavior track the information the claimed reasoning capability should depend on? This broader scope also means that intervention design must remain domain-specific rather than treating ARC-style transformations as a universal template.

The resulting perspective is intentionally narrower than a complete theory of reasoning. It asks whether a claimed reasoning capability exhibits predictable behavioral dependencies on the task structure and evidence it should use. This perspective does not resolve every ambiguity about internal reasoning, but it makes claims about abstraction transfer and hypothesis revision more explicit, testable, and falsifiable.

## Conclusion

Reasoning evaluation now extends beyond final-answer accuracy to reasoning traces, process quality, test-time resources, and human-relative eficiency. These measures remain valuable, but they do not establish whether reasoning adapts when relevant task structure or available evidence changes.

We characterized adaptive reasoning at two complementary scales. Across-task adaptivity tests whether acquired abstractions transfer or recombine under controlled task variation, while within-task adaptivity tests whether intermediate decisions revise in response to new observations, failures, or feedback. Using ARC-AGI as a diagnostic case study, we showed how these distinctions motivate paired interventionoriented comparisons with explicit reporting of test-time budgets and observable decisions.

Adaptive evidence does not certify alignment or reveal a model’s complete internal mechanism. Its value is narrower but consequential: it makes claims about transfer and revision more directly testable, falsifiable, and auditable. Reasoning claims require evidence about what a system produces and how its behavior changes when information relevant to its reasoning is deliberately changed.

## References

Akyürek, E.; Damani, M.; Zweiger, A.; Qiu, L.; Guo, H.; Pari, J.; Kim, Y.; and Andreas, J. 2025. The Surprising Efectiveness of Test-Time Training for Few-Shot Learning. In ICML.

ARC Foundation. 2026. ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence. arXiv:2603.24621.

Atıl, B.; Aykent, S.; Chittams, A.; Fu, L.; Passonneau, R. J.; Radclife, E.; Rajagopal, G. R.; Sloan, A.; Tudrej, T.; Türe, F.; Wu, Z.; Xu, L.; and Baldwin, B. 2025. Non-Determinism of “Deterministic” LLM System Settings in Hosted Environments. In AACL Eval4NLP Workshop.

Brown, B.; Juravsky, J.; Ehrlich, R.; Clark, R.; Le, Q. V.; Ré, C.; and Mirhoseini, A. 2024. Large Language Monkeys: Scaling Inference Compute with Repeated Sampling. arXiv:2407.21787.

Chollet, F. 2019. On the Measure of Intelligence. arXiv:1911.01547.

Chollet, F.; Knoop, M.; Kamradt, G.; Landers, B.; and Pinkard, H. 2025. ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems. arXiv:2505.11831.

Gardner, M.; Artzi, Y.; Basmov, V.; Berant, J.; Bogin, B.; Chen, S.; Dasigi, P.; Dua, D.; Elazar, Y.; Gottumukkala, A.; Gupta, N.; Hajishirzi, H.; Ilharco, G.; Khashabi, D.; Lin, K.; Liu, J.; Liu, N. F.; Mulcaire, P.; Ning, Q.; Singh, S.; Smith, N. A.; Subramanian, S.; Tsarfaty, R.; Wallace, E.; Zhang, A.; and Zhou, B. 2020. Evaluating Models’ Local Decision Boundaries via Contrast Sets. In EMNLP Findings.

Hadfield-Menell, D.; Dragan, A.; Abbeel, P.; and Russell, S. 2017. The Of-Switch Game. In IJCAI.

Hwang, S.; Lee, S.; Kim, S.; and Kim, S. 2025. Solution Augmentation for ARC-AGI Problems Using GFlowNet: A Probabilistic Exploration Approach. Transactions on Machine Learning Research.

Jeon, J.; Shim, S.; Kim, S.; Kim, S.; and Lee, B.-J. 2026. Neural MCTS with LLM Guidance for Efective Program Synthesis on Abstraction and Reasoning Corpus. Machine Learning, 115(8): 181.

Kim, S.; Choi, H.; Lee, S.; and Kim, S. 2026. ARCTraj: A Dataset and Benchmark of Human Reasoning Trajectories for Abstract Problem Solving. In KDD.

Kim, S.; and Kim, S. 2024. System-2 Reasoning via Generality and Adaptation. In NeurIPS S2R Workshop.

Kim, S.; Lee, H.; and Kim, S. 2025. Addressing and Visualizing Misalignments in Human Task-Solving Trajectories. In KDD.

Lee, S.; Sim, W.; Shin, D.; Seo, W.; Park, J.; Lee, S.; Hwang, S.; Kim, S.; and Kim, S. 2025. Reasoning Abilities of Large Language Models: In-Depth Analysis on the Abstraction and Reasoning Corpus. ACM Transactions on Intelligent Systems and Technology, 16(6): 1–52.

LeGris, S.; Vong, W. K.; Lake, B. M.; and Gureckis, T. M. 2025. A Comprehensive Behavioral Dataset for the Abstraction and Reasoning Corpus. Scientific Data, 12: 1380.

Lightman, H.; Kosaraju, V.; Burda, Y.; Edwards, H.; Baker, B.; Lee, T.; Leike, J.; Schulman, J.; Sutskever, I.; and Cobbe, K. 2024. Let’s Verify Step by Step. In ICLR.

Madaan, A.; Tandon, N.; Gupta, P.; Hallinan, S.; Gao, L.; Wiegrefe, S.; Alon, U.; Dziri, N.; Prabhumoye, S.; Yang, Y.; Gupta, S.; Prasad Majumder, B.; Hermann, K.; Welleck, S.; Yazdanbakhsh, A.; and Clark, P. 2023. Self-Refine: Iterative Refinement with Self-Feedback. In NeurIPS.

Moskvichev, A.; Odouard, V. V.; and Mitchell, M. 2023. The ConceptARC Benchmark: Evaluating Understanding and Generalization in the ARC Domain. Transactions on Machine Learning Research.

Pourcel, J.; Colas, C.; and Oudeyer, P.-Y. 2025. Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI. In ICML.

Ribeiro, M. T.; Wu, T.; Guestrin, C.; and Singh, S. 2020. Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. In ACL.

Shinn, N.; Cassano, F.; Gopinath, A.; Narasimhan, K.; and Yao, S. 2023. Reflexion: Language Agents with Verbal Reinforcement Learning. In NeurIPS.

Snell, C.; Lee, J.; Xu, K.; and Kumar, A. 2025. Scaling LLM Test-Time Compute Optimally Can be More Efective than Scaling Parameters for Reasoning. In ICLR.

Soares, N.; Fallenstein, B.; Yudkowsky, E.; and Armstrong, S. 2015. Corrigibility. In AAAI AI&Ethics Workshop.

Turpin, M.; Michael, J.; Perez, E.; and Bowman, S. 2023. Language Models Don’t Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. In NeurIPS.

Uesato, J.; Kushman, N.; Kumar, R.; Song, F.; Siegel, N.; Wang, L.; Creswell, A.; Irving, G.; and Higgins, I. 2022. Solving Math Word Problems with Process- and Outcome-Based Feedback. arXiv:2211.14275.

Wang, R.; Zelikman, E.; Poesia, G.; Pu, Y.; Haber, N.; and Goodman, N. 2024. Hypothesis Search: Inductive Reasoning with Language Models. In ICLR.

Wang, X.; Wei, J.; Schuurmans, D.; Le, Q. V.; Chi, E. H.; Narang, S.; Chowdhery, A.; and Zhou, D. 2023. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In ICLR.

Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Ichter, B.; Xia, F.; Chi, E. H.; Le, Q. V.; and Zhou, D. 2022. Chainof-Thought Prompting Elicits Reasoning in Large Language Models. In NeurIPS.

Zhang, D.; Li, Z.-Z.; Zhang, M.-L.; Zhang, J.; Liu, Z.; Yao, Y.; Xu, H.; Zheng, J.; Chen, X.; Zhang, Y.; Yin, F.; Dong, J.; Guo, Z.; Song, L.; and Liu, C.-L. 2026. From System 1 to System 2: A Survey of Reasoning Large Language Models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(3): 3335–3354.

## Technical Supplement

This supplementary material provides additional positioning and worked examples for the intervention-oriented evaluation framework introduced in the main paper. The examples illustrate how across-task abstraction transfer and within-task evidencesensitive revision can be operationalized under explicit test-time resource reporting. They are intended as demonstrations of the evaluation procedure rather than as a new ARC-AGI benchmark or a comprehensive model comparison.

## Relation to Closely Related Evaluation Paradigms

Several existing evaluation methods use controlled variations of an input or task to examine how model behavior changes Contrast sets construct minimally diferent examples to probe local decision boundaries (Gardner et al. 2020). CheckList uses behavioral tests designed around specific capabilities rather than aggregate accuracy alone (Ribeiro et al. 2020). In the ARC domain, ConceptARC groups tasks by human-interpretable concepts and evaluates whether models generalize across tasks that share those concepts (Moskvichev, Odouard, and Mitchell 2023). Our across-task evaluation follows a related idea, but focuses on whether behavior changes consistently with the task structure that an intervention preserves or modifies.

Belief revision is closely related to our definition of within-task adaptivity. Belief-R tests whether language models revise a prior conclusion when additional premises warrant a change, and whether they retain the conclusion when the new information does not warrant one (Wilie et al. 2024). Our within-task evaluation uses the same basic distinction. Contradictory evidence should change the part of a hypothesis that it bears on, while confirming or irrelevant evidence should not lead to unrelated changes. We apply this logic to observable intermediate decisions more broadly, including hypotheses, programs, plans, actions, and tool calls.

Counterfactual interventions have also been used to test whether intermediate reasoning afects later reasoning or final answers (Xiong et al. 2025). These studies focus on whether changing a reasoning artifact changes subsequent computation. Our evaluation asks a diferent question: whether a change in task-relevant information produces the corresponding change in reasoning or behavior. These properties need not coincide. A reasoning trace may influence the final answer without being revised when corrective evidence is introduced. Likewise, a diferent final answer does not by itself show that the part of the hypothesis contradicted by the new evidence was revised.

The novelty of our framework is therefore not the use of controlled interventions itself. We organize adaptive reasoning into two evaluation settings, across-task abstraction transfer and within-task evidence-sensitive revision, and evaluate both by relating controlled changes in task structure or evidence to subsequent behavior. We also make the available test-time resources explicit so that evidence-sensitive revision can be distinguished from gains obtained through additional search or sampling.

<table><tr><td>Approach</td><td>Variation or intervention</td><td>Main evaluative target</td><td>Connection to adaptive reasoning</td></tr><tr><td>Contrast Sets</td><td>Local changes to an input or task instance</td><td>Sensitivity of model predictions to meaningful local contrasts</td><td>Provides a basis for constructing controlled task variants, which we use to test whether behavior follows preserved or changed task structure.</td></tr><tr><td>CheckList</td><td>Behavioral test conditions and targeted input perturbations</td><td>Capability-specific behavioral testing</td><td>Motivates targeted tests and controls that isolate the behavior associated with a specified capability.</td></tr><tr><td>ConceptARC</td><td>Concept-centered families of ARC tasks</td><td>Generalization across tasks that share a concept</td><td>Closely related to across-task adaptivity, which additionally distinguishes concept-preserving from</td></tr><tr><td>Belief-R</td><td>Newly supplied textual evidence</td><td>Revision or maintenance of a prior conclusion</td><td>concept-changing interventions. Closely related to within-task adaptivity, which applies the same distinction to observable decisions beyond textual conclusions.</td></tr><tr><td>Thinking-draft faithfulness</td><td>Intermediate reasoning content</td><td>Causal influence of reasoning on later reasoning or answers</td><td>Tests dependence on intervened reasoning content, whereas our within-task evaluation changes task-relevant evidence and examines</td></tr><tr><td>Our framework</td><td>Task structure or task-specific evidence</td><td>Abstraction transfer and evidence-sensitive revision</td><td>the resulting revision. Evaluates across-task and within-task adaptivity under controlled changes while making test-time resources explicit.</td></tr></table>

Table 3: Relationship to closely related evaluation paradigms. The comparison focuses on the information varied by each approach and the behavioral question it is designed to answer.

## Worked Example of Across-Task Adaptivity

Across-task adaptivity asks whether behavior changes in a way that matches controlled changes in the task structure relevant to a reasoning claim. Let $C ( \tau )$ denote the evaluator-specified structure of task τ. For an intervention $I _ { T }$ producing $\tau ^ { \prime } = I _ { T } ( \tau )$ the main paper defines

$$
\begin{array} { r l } & { \Delta C = C ( \tau ^ { \prime } ) \ominus C ( \tau ) , } \\ & { \Delta B = B _ { \theta } ( \tau ^ { \prime } , E _ { 0 } ; q ) \ominus B _ { \theta } ( \tau , E _ { 0 } ; q ) . } \end{array}
$$

The question is whether $\Delta C \sim \Delta B$ under a task-appropriate consistency relation. A concept-preserving intervention should preserve the behavior associated with the target structure, possibly under a predictable transformation, whereas a conceptchanging intervention should induce a corresponding change in behavior.

Downward Relocation in ARC-AGI-1 Fig. 3 uses three related tasks from ARC-AGI-1, 1e0a9b12, 3618c87e, and 25ff71a9, to illustrate this distinction (ARC Prize Foundation 2026). These tasks form an illustrative comparison rather than a synthetically controlled intervention set. The evaluator therefore specifies the relevant task structure explicitly before comparing them.

![](images/1deac551dd2877037963389d09bbc0bb2ff8e136f59a7fcbdb6ab15e9d5ec279.jpg)

![](images/c7fb3da37ad15cd21984c62ec5db300968e6a36698b811655bd57b4269ac6210.jpg)

![](images/5db7b14e4ecc740981e23fd2d8f2c14d1d5c5dd1132464f2053c9f731b52bf0b.jpg)  
Figure 3: Across-task adaptivity illustrated with related ARC-AGI-1 tasks. The base task, 1e0a9b12, applies pixel-level downward relocation. Under the evaluator-specified abstraction, 3618c87e is concept-preserving: the manipulation unit and downward-relocation rule remain unchanged despite the introduction of a structured gray background. In 25ff71a9, the manipulation unit changes from individual pixels to a connected object, yielding an object-level downward-relocation rule and a concept-changing variant.

In the base task, 1e0a9b12, colored pixels are relocated downward within their columns. We define the structure relevant to this example as pixel-level downward relocation. The particular colors and initial positions of the pixels are not part of thi abstraction.

Task 3618c87e introduces a structured gray background, but the movable elements remain individual pixels and the relevant operation remains downward relocation. Under this evaluator-specified abstraction, we therefore treat

$$
C ( \tau _ { 3 6 1 8 \mathrm { c } 8 7 \mathrm { e } } ) = C ( \tau _ { 1 \mathrm { e } 0 \bar { \mathsf { a } } 9 \mathrm { b } 1 2 } ) .
$$

The task serves as a concept-preserving example because the additional background structure changes the visual context without changing the manipulation unit or the downward-relocation rule.

Task 25ff71a9 retains the general downward-relocation motif but changes the manipulation unit. Connected colored pixels must be treated as a single object whose internal configuration is preserved during relocation. We therefore distinguish its task structure from the pixel-level rule:

$$
C ( \tau _ { 2 5 \tt f f 7 1 a 9 } ) \not = C ( \tau _ { 1 \tt e 0 a 9 b 1 2 } ) .
$$

This provides a concept-changing example in which the relevant structural diference is the shift from pixel-level to object-level manipulation.

The comparison illustrates why across-task adaptivity is not equivalent to invariance. A solver that has captured the pixel level abstraction should remain efective when the surrounding background structure changes, as in 3618c87e. In contrast, 25ff71a9 requires behavior appropriate to an object-level representation. The evaluator can therefore distinguish changes that preserve the target abstraction from changes that alter a component of the task structure on which the reasoning claim depends.

## Worked Example of Within-Task Adaptivity

Within-task adaptivity asks whether new evidence changes an observable intermediate decision $Z _ { t }$ and subsequent behavior. For evidence $E _ { t }$ and an intervention $E _ { t } ^ { \prime } = I _ { E } ( E _ { t } )$ , the main paper considers

$$
\Delta E \longrightarrow \Delta Z _ { t } \longrightarrow \Delta B _ { > t } .
$$

The key question is whether the change corresponds to what the new evidence supports or rules out.

Interaction Evidence in ARC-AGI-3 Consider ARC-AGI-3 task r11L (ARC Prize Foundation 2026). In Level 1, a purple object must be moved into a target by manipulating two connected endpoints. Clicking a black cell moves the white endpoint, while clicking the dark-gray endpoint swaps the white and dark-gray roles. Light-gray cells are obstacles, and contact between the purple object and an obstacle causes failure.

(a) Initial state  
![](images/ccd130b79a0ab3e597e635d7f9158a725bce6ce5fe2d853f55992a12348efe61.jpg)

(b) Confirming evidence  
![](images/3147262c46b164794a494ed34a21a24b1cb0cfc66afaf2f242c7d258cd235372.jpg)  
Lv.1 begins with the endpoint-control mechanism still to be inferred.  
Clicking a black cell moves the current white endpoint to the selected location.

(c) Contradictory evidence  
(d) Next level  
![](images/532868c8bcc43aa5f770fd626d25f806b045b2a7ed7a1c9c37b7cfc03ba61a36.jpg)  
Clicking the dark-gray endpoint swaps the white and dark-gray roles.

![](images/1ca9bd9193da479484d6f3d9f7a1ede9c1098aaf153a086594da137a2fa96038.jpg)  
Lv.2 introduces new structures while previously learned interaction rules remain relevant.  
Figure 4: Within-task adaptivity in ARC-AGI-3 task r11L. (a) The initial state of Level 1. (b) Clicking a black cell moves the current white endpoint, providing confirming evidence about endpoint control. (c) Clicking the dark-gray endpoint swaps the white and dark-gray roles, contradicting the hypothesis that the dark-gray endpoint is fixed and motivating a targeted revision. (d) After Level 1 is cleared, Level 2 introduces additional structures while previously acquired interaction rules remain useful. The sequence illustrates how supported knowledge can be retained while task-specific hypotheses are revised or extended as new evidence becomes available.

Fig. 4 shows how interaction reveals these rules and how they remain relevant when new structures appear in Level 2.

Suppose the agent holds $H _ { 1 } ;$ “The white endpoint is movable, whereas the dark-gray endpoint is fixed.” A black-cell click supports the first component, while a role swap contradicts the second. To test whether the resulting update is evidence-sensitive, the evaluator can branch from the same pre-intervention state and hypothesis, as summarized in Table 4.

<table><tr><td>Branch</td><td>Evidence</td><td>Expected response</td></tr><tr><td>Contradictory</td><td>Clicking the dark-gray endpoint swaps the white Revise the claim that the dark-gray endpoint is and dark-gray roles.</td><td>fixed.</td></tr><tr><td>Confirming</td><td>Clicking à black cell moves the white endpoint to that cell.</td><td>Retain the claim that the white endpoint is mov- able.</td></tr><tr><td>Irrelevant</td><td>Clicking a light-gray obstacle produces no end- Preserve the endpoint-role hypothesis. point movement.</td><td></td></tr><tr><td>No added evidence</td><td>No diagnostic interaction occurs</td><td>Continue under the same test-time budget</td></tr></table>

Table 4: Same-history branching in ARC-AGI-3 task r11L. Each branch specifies the expected response to diferent evidence about the current hypothesis.

The role-swap observation provides evidence for revision only if the agent updates its endpoint-control hypothesis and subsequent actions reflect the new rule. An agent may instead continue trying diferent clicks until it succeeds, which demonstrates search rather than evidence-sensitive revision. Evaluation should therefore record $Z _ { t }$ before and after the diagnostic interaction, subsequent behavior, and the test-time budget.

## Operationalizing the Framework

The framework leaves some quantities domain-specific because the relevant task structure and observable behavior depend on the setting. To avoid post-hoc interpretation, the evaluator should specify what is changed, what is preserved, and what response is expected before examining the model’s response to the intervention. Tab. 5 summarizes the information needed for across-task and within-task evaluation.

<table><tr><td>Stage</td><td>Across-task</td><td>Within-task</td></tr><tr><td>Before intervention</td><td>Specify the target structure  $C ( \tau )$  to be preserved.</td><td>and the properties Record the available evidence  $E _ { t }$  and an observable decision  $Z _ { t } ,$  such as a hypothesis, program, plan, action, or tool call.</td></tr><tr><td>Intervention</td><td>Modify a surface property or the target structure Add or modify evidence to produce while controlling other relevant properties.</td><td> $E _ { t } ^ { \prime }$  and identify which part of  $\scriptstyle \sum _ { t }$  the new evidence bears on.</td></tr><tr><td>Expected response</td><td>transform predictably, or change with the modified revised or retained given the new evidence. structure.</td><td>Specify whether behavior should remain stable, Specify whether the implicated decision should be</td></tr><tr><td>Observed response</td><td>Compare behavior across the paired task variants.</td><td>Compare the decision before and after the interven- tion and record subsequent behavior  $B _ { > t }$ </td></tr><tr><td>Controls</td><td></td><td>Use matched variants and report the test-time budget. Use no-intervention or irrelevant-evidence branches when possible, and report the test-time budget.</td></tr></table>

Table 5: Operationalization of across-task and within-task adaptivity.

The relevance, specificity, and consistency criteria defined in the main paper are used to interpret these comparisons. Together, they test whether the observed response is tied to the information changed by the intervention and whether similar interventions produce similar response patterns.

For stochastic systems, repeated runs can help separate intervention-related changes from ordinary sampling variation. Testtime resources should be held fixed or reported explicitly so that additional search or sampling is not mistaken for adaptation. Across-task and within-task results should be reported separately because they capture diferent forms of adaptive behavior.

## Intervention Validity and Limitations

The interpretation of an intervention depends on whether it changes the intended property while leaving other relevant properties unchanged. A concept-preserving intervention may inadvertently alter the target structure, while a concept-changing intervention may modify several relevant factors at once. Interventions should therefore be kept as local as possible, with the changed and preserved properties specified explicitly.

Concept-preserving interventions. Consider a task in which color determines the semantic role of an object. Permuting the colors would not be a surface-level change because it could also change the target structure C(τ). Such a variant should not be treated as a test of invariance to color.

Concept-changing interventions. Consider a variant that changes the target spatial relation together with the number, identity, and layout of the objects. If the model’s behavior changes, it is dificult to determine which modification caused the response. A more informative comparison changes the target relation while keeping the remaining task properties fixed whenever possible.

Limits of behavioral evidence. A consistent pattern of transfer or revision does not identify the internal mechanism that produces it. Diferent mechanisms may lead to similar observed behavior, and an observable decision $Z _ { t }$ may capture only part of the model’s internal reasoning. The resulting evidence therefore supports claims about behavioral transfer and revision, not about a unique internal mechanism.

## Reporting Checklist

When reporting an intervention-based evaluation, we recommend including:

1. the reasoning claim, the relevant task structure or evidence, and what the intervention is intended to change or preserve;

2. the expected response specified before examining the model output, together with the observed task outcome and intermediate decision when available;

3. the test-time budget and the controls used to distinguish the efect of the intervention from additional search, sampling, or unrelated variation; and

4. repeated runs when needed for stochastic systems, with across-task and within-task results reported separately.

These items are intended to make claims about abstraction transfer and evidence-sensitive revision easier to interpret and compare. They do not identify a unique internal reasoning mechanism or establish that a system is aligned.

Hajishirzi, H.; Ilharco, G.; Khashabi, D.; Lin, K.; Liu, J.; Liu, N. F.; Mulcaire, P.; Ning, Q.; Singh, S.; Smith, N. A.; Subramanian,

## Supplementary References

Gardner, M.; Artzi, Y.; Basmov, V.; Berant, J.; Bogin, B.; Chen, S.; Dasigi, P.; Dua, D.; Elazar, Y.; Gottumukkala, A.; Gupta, N.;

S.; Tsarfaty, R.; Wallace, E.; Zhang, A.; and Zhou, B. 2020. Evaluating Models’ Local Decision Boundaries via Contrast Sets. In EMNLP Findings.

Ribeiro, M. T.; Wu, T.; Guestrin, C.; and Singh, S. 2020. Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. In ACL.

Moskvichev, A.; Odouard, V. V.; and Mitchell, M. 2023. The ConceptARC Benchmark: Evaluating Understanding and Generalization in the ARC Domain. TMLR.

Wilie, B.; Cahyawijaya, S.; Ishii, E.; He, J.; and Fung, P. 2024. Belief Revision: The Adaptability of Large Language Models Reasoning. In EMNLP.

Xiong, Z.; Chen, S.; Qi, Z.; and Lakkaraju, H. 2025. Measuring the Faithfulness of Thinking Drafts in Large Reasoning Models. In NeurIPS.

ARC Prize Foundation. 2026. Dataset: ARC-AGI Tasks. https://arcprize.org/tasks